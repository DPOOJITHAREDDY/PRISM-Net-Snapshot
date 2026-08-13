#!/usr/bin/env python
"""
train.py

PRISM-Net training entry point.

Example:
    python train.py ^
        --data_root "C:\\Users\\Poojitha Reddy\\Downloads\\train\\train" ^
        --output_dir "checkpoints" ^
        --epochs 50 ^
        --batch_size 8 ^
        --lr 0.0002 ^
        --device cuda

To resume an interrupted run:
    python train.py --resume checkpoints\\best_model.pth --data_root ... --epochs 50 ...
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.augmentations import (
    ComposeTrainTransform,
    CurriculumDegradationAugment,
    CurriculumSchedule,
    PairedGeometricAugment,
)
from src.dataset import PrismNetTrainDataset, make_train_val_split
from src.losses import CompositeLoss
from src.metrics import compute_psnr, compute_ssim
from src.model import PRISMNet
from validate_dataset import ValidationError, validate_train_root


def check_tensor_finiteness(name: str, tensor: torch.Tensor, print_stats: bool = True) -> bool:
    """
    Check a tensor for NaN/Inf and optionally print min/max/mean.
    Returns True if finite, False if any non-finite value found.
    """
    if tensor is None:
        return True
    is_finite = torch.isfinite(tensor).all().item()
    if not is_finite or print_stats:
        shape_str = f"shape={tuple(tensor.shape)}"
        if print_stats and is_finite:
            min_v = tensor.min().item()
            max_v = tensor.max().item()
            mean_v = tensor.mean().item()
            print(f"  [{name}] {shape_str} min={min_v:.4f} max={max_v:.4f} mean={mean_v:.4f}")
        else:
            print(f"  [{name}] {shape_str} CONTAINS NON-FINITE VALUES")
            if tensor.dim() > 0 and tensor.numel() < 100:
                print(f"      content: {tensor}")
    return is_finite


def parse_args():
    p = argparse.ArgumentParser(description="Train PRISM-Net.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--log_dir", type=str, default="experiments/tensorboard")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noisy_crop_size", type=int, default=96)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--skip_validation_gate", action="store_true",
                    help="Skip the pre-training validate_dataset.py check (not recommended).")
    p.add_argument("--warmup_steps", type=int, default=500,
                    help="Linear LR warmup steps within epoch 0 only. Set 0 to disable.")
    p.add_argument("--max_nan_batch_fraction", type=float, default=0.5,
                    help="Abort training if more than this fraction of an epoch's "
                         "batches produce a non-finite loss.")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--num_denoise_blocks", type=int, default=8)
    p.add_argument("--num_refine_blocks", type=int, default=4)
    p.add_argument("--cond_dim", type=int, default=64)
    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_psnr, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_psnr": best_val_psnr,
            "args": vars(args),
        },
        path,
    )


@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    psnrs, ssims = [], []
    for batch in val_loader:
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)
        pred = model(noisy)
        for i in range(pred.shape[0]):
            psnrs.append(compute_psnr(pred[i], gt[i]))
            ssims.append(compute_ssim(pred[i], gt[i]))
    model.train()
    return {"val_psnr": sum(psnrs) / len(psnrs), "val_ssim": sum(ssims) / len(ssims)}


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- Hard validation gate (Phase 3) before anything else runs ---
    if not args.skip_validation_gate:
        try:
            validate_train_root(args.data_root, full=False)
        except ValidationError as exc:
            print(f"\nTRAINING ABORTED -- dataset validation failed: {exc}", file=sys.stderr)
            sys.exit(1)

    device = args.device
    print(f"Using device: {device}")

    # --- Resolve architecture: checkpoint's saved args win on resume ---
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        saved_args = resume_ckpt.get("args", {})
        width = saved_args.get("width", args.width)
        num_denoise_blocks = saved_args.get("num_denoise_blocks", args.num_denoise_blocks)
        num_refine_blocks = saved_args.get("num_refine_blocks", args.num_refine_blocks)
        cond_dim = saved_args.get("cond_dim", args.cond_dim)
        if any(saved_args.get(k) != getattr(args, k) for k in
               ("width", "num_denoise_blocks", "num_refine_blocks", "cond_dim") if k in saved_args):
            print("NOTE: using architecture hyperparameters from the checkpoint "
                  "(may differ from the --width/--num_denoise_blocks/... flags passed this run).")
    else:
        width = args.width
        num_denoise_blocks = args.num_denoise_blocks
        num_refine_blocks = args.num_refine_blocks
        cond_dim = args.cond_dim

    # --- Data ---
    train_stems, val_stems = make_train_val_split(
        args.data_root, val_fraction=args.val_fraction, seed=args.split_seed
    )
    print(f"Train pairs: {len(train_stems)}  |  Val pairs: {len(val_stems)}  (seed={args.split_seed})")

    geometric = PairedGeometricAugment(noisy_crop_size=args.noisy_crop_size)
    degradation_aug = CurriculumDegradationAugment(CurriculumSchedule())
    train_transform = ComposeTrainTransform(geometric, degradation_aug)

    train_ds = PrismNetTrainDataset(args.data_root, stems=train_stems, transform=train_transform)
    val_ds = PrismNetTrainDataset(args.data_root, stems=val_stems, transform=None)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.startswith("cuda")), drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- Model / loss / optimizer ---
    model = PRISMNet(
        width=width,
        num_denoise_blocks=num_denoise_blocks,
        num_refine_blocks=num_refine_blocks,
        cond_dim=cond_dim,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    loss_fn = CompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.startswith("cuda")))

    start_epoch = 0
    best_val_psnr = -float("inf")

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        if resume_ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_psnr = resume_ckpt.get("best_val_psnr", -float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_val_psnr={best_val_psnr:.3f}")

    writer = SummaryWriter(log_dir=str(log_dir))
    history = []

    for epoch in range(start_epoch, args.epochs):
        train_transform.set_epoch(epoch, args.epochs)
        stage = degradation_aug.stage
        print(f"\nEpoch {epoch+1}/{args.epochs} -- curriculum: "
              f"speckle={stage.speckle_severity:.3f} gaussian={stage.gaussian_severity:.3f} "
              f"blur={stage.blur_severity:.3f} apply_prob={stage.apply_prob:.3f}")

        model.train()
        epoch_start = time.time()
        running = {"total": 0.0, "charbonnier": 0.0, "ms_ssim_loss": 0.0, "sobel": 0.0, "fft": 0.0}
        n_batches = 0
        n_skipped = 0
        n_total_batches_seen = 0

        for batch_idx, batch in enumerate(train_loader):
            n_total_batches_seen += 1
            noisy = batch["noisy"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            # Linear LR warmup, epoch 0 only.
            if epoch == 0 and args.warmup_steps > 0 and batch_idx < args.warmup_steps:
                warmup_lr = args.lr * (batch_idx + 1) / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            optimizer.zero_grad(set_to_none=True)

            # ==== DIAGNOSTIC: check input before forward pass ====
            if not hasattr(check_tensor_finiteness, "_first_nan_found"):
                if not check_tensor_finiteness("augmented noisy input", noisy, print_stats=True):
                    check_tensor_finiteness._first_nan_found = True
                    print(f"\n!!! FIRST NaN FOUND at epoch {epoch+1}, batch {batch_idx} in augmented input !!!")
                    print(f"    Curriculum stage: speckle={stage.speckle_severity:.5f}, "
                          f"gaussian={stage.gaussian_severity:.5f}, blur={stage.blur_severity:.5f}, "
                          f"apply_prob={stage.apply_prob:.3f}")
                    n_skipped += 1
                    continue

            # Disable autocast for model forward: numerically fragile ops
            # (log in preprocessing, LayerNorm variance) are unstable in fp16.
            # Backward pass still uses GradScaler for speed.
            pred = model(noisy)

            # ==== DIAGNOSTIC: check model output before loss ====
            if not hasattr(check_tensor_finiteness, "_first_nan_found"):
                if not check_tensor_finiteness("model prediction output", pred, print_stats=True):
                    check_tensor_finiteness._first_nan_found = True
                    print(f"\n!!! FIRST NaN FOUND at epoch {epoch+1}, batch {batch_idx} in model output !!!")
                    print(f"    Input was finite; NaN originated in PRISM-Net forward pass.")
                    n_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

            # Loss computed in float32, OUTSIDE autocast.
            total_loss, components = loss_fn(pred.float(), gt.float())

            # ==== DIAGNOSTIC: check loss components ====
            if not hasattr(check_tensor_finiteness, "_first_nan_found"):
                if not torch.isfinite(total_loss):
                    check_tensor_finiteness._first_nan_found = True
                    print(f"\n!!! FIRST NaN FOUND at epoch {epoch+1}, batch {batch_idx} in loss computation !!!")
                    for k, v in components.items():
                        status = "OK" if isinstance(v, float) and -1e6 < v < 1e6 else "NON-FINITE"
                        print(f"    {k}: {v} ({status})")
                    print(f"    Prediction and GT were both finite; NaN originated in loss function.")
                    n_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

            if not torch.isfinite(total_loss):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            for k in running:
                running[k] += components[k]
            n_batches += 1

        if n_total_batches_seen > 0 and (n_skipped / n_total_batches_seen) > args.max_nan_batch_fraction:
            print(
                f"\nTRAINING ABORTED at epoch {epoch+1}: {n_skipped}/{n_total_batches_seen} "
                f"batches ({n_skipped / n_total_batches_seen:.0%}) produced a non-finite loss.",
                file=sys.stderr,
            )
            sys.exit(1)

        if n_batches == 0:
            print(
                f"\nTRAINING ABORTED at epoch {epoch+1}: every batch this epoch was skipped.",
                file=sys.stderr,
            )
            sys.exit(1)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg = {k: v / n_batches for k, v in running.items()}
        print(f"  train loss: total={avg['total']:.4f} charbonnier={avg['charbonnier']:.4f} "
              f"ms_ssim={avg['ms_ssim_loss']:.4f} sobel={avg['sobel']:.4f} fft={avg['fft']:.4f} "
              f"({epoch_time:.1f}s"
              + (f", {n_skipped} batch(es) skipped" if n_skipped else "") + ")")

        for k, v in avg.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/n_skipped_batches", n_skipped, epoch)

        val_metrics = None
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = run_validation(model, val_loader, device)
            print(f"  val: PSNR={val_metrics['val_psnr']:.3f}  SSIM={val_metrics['val_ssim']:.4f}")
            writer.add_scalar("val/psnr", val_metrics["val_psnr"], epoch)
            writer.add_scalar("val/ssim", val_metrics["val_ssim"], epoch)

            if val_metrics["val_psnr"] > best_val_psnr:
                best_val_psnr = val_metrics["val_psnr"]
                save_checkpoint(output_dir / "best_model.pth", model, optimizer, scheduler, scaler,
                                 epoch, best_val_psnr, args)
                print(f"  -> new best model saved (val PSNR={best_val_psnr:.3f})")

        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, scaler,
                         epoch, best_val_psnr, args)

        history.append({"epoch": epoch, "train": avg, "val": val_metrics,
                         "epoch_time_sec": epoch_time, "n_skipped_batches": n_skipped})
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    writer.close()
    print(f"\nTraining complete. Best val PSNR: {best_val_psnr:.3f}")
    print(f"Checkpoints: {output_dir / 'best_model.pth'}, {output_dir / 'last.pth'}")


if __name__ == "__main__":
    main()