#!/usr/bin/env python
"""
tests/smoke_test.py

End-to-end smoke test suite for PRISM-Net, run before committing to a
full training job. Exercises the entire pipeline on real data (a
handful of samples, not the full 3,200) plus one optimizer step
(verified to actually change model weights) and a checkpoint
save/reload round-trip (verified to produce identical outputs).

Usage:
    python tests\\smoke_test.py --data_root "C:\\...\\train\\train" --test_dir "C:\\...\\Test_NoisyLR"

Exits 0 on full success, 1 on any failure, with a clear message
identifying which check failed.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.dataset import PrismNetTestDataset, PrismNetTrainDataset, make_train_val_split
from src.losses import CompositeLoss
from src.metrics import compute_psnr, compute_ssim
from src.model import PRISMNet

_state = {}  # small stash for values shared between check steps


def check(name, fn):
    print(f"[{name}] running...")
    try:
        fn()
    except Exception as exc:
        print(f"[{name}] FAILED: {exc}", file=sys.stderr)
        raise
    print(f"[{name}] OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # 1. import all modules -- already exercised by the imports above.
    check("1_import_modules", lambda: None)

    # 2. load one training pair
    train_stems, val_stems = make_train_val_split(args.data_root, val_fraction=0.1, seed=42)
    train_ds = PrismNetTrainDataset(args.data_root, stems=train_stems[:8])

    def _load_train_pair():
        sample = train_ds[0]
        assert sample["noisy"].shape == (1, 128, 128)
        assert sample["gt"].shape == (1, 256, 256)

    check("2_load_training_pair", _load_train_pair)

    # 3. load one test sample
    test_ds = PrismNetTestDataset(args.test_dir)

    def _load_test_sample():
        sample = test_ds[0]
        assert sample["noisy"].shape == (1, 128, 128)

    check("3_load_test_sample", _load_test_sample)

    # 4 & 5. forward pass + output dimensions
    model = PRISMNet().to(args.device)

    def _forward_pass():
        x = torch.stack([train_ds[i]["noisy"] for i in range(4)]).to(args.device)
        out = model(x)
        assert out.shape == (4, 1, 256, 256), f"got {out.shape}"

    check("4_5_forward_pass_output_dims", _forward_pass)

    # 6. loss calculation
    loss_fn = CompositeLoss().to(args.device)

    def _loss_calc():
        x = torch.stack([train_ds[i]["noisy"] for i in range(4)]).to(args.device)
        y = torch.stack([train_ds[i]["gt"] for i in range(4)]).to(args.device)
        pred = model(x)
        total, components = loss_fn(pred, y)
        assert torch.isfinite(total)
        _state["loss"] = total

    check("6_loss_calculation", _loss_calc)

    # 7. optimizer step -- and verify at least one parameter genuinely
    # changed, not just that .step() ran without raising.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def _optimizer_step():
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.zero_grad()
        _state["loss"].backward()
        optimizer.step()
        after = list(model.parameters())
        total_change = sum((b - a.detach()).abs().sum().item() for b, a in zip(before, after))
        assert total_change > 0, "no model parameters changed after the optimizer step"
        _state["param_change_total"] = total_change

    check("7_optimizer_step_changes_parameters", _optimizer_step)
    print(f"    (total absolute parameter change across the model: {_state['param_change_total']:.6f})")

    # 8 & 9. checkpoint save + reload, verified to give identical outputs
    def _checkpoint_roundtrip():
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "smoke_test.pth"
            torch.save({"model_state": model.state_dict(), "args": {
                "width": 48, "num_denoise_blocks": 8, "num_refine_blocks": 4, "cond_dim": 64,
            }}, ckpt_path)

            reloaded = PRISMNet().to(args.device)
            # weights_only=False: this checkpoint stores a plain "args"
            # dict alongside the tensors -- torch>=2.6 defaults
            # weights_only=True, which can reject that.
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            reloaded.load_state_dict(ckpt["model_state"])

            model.eval()
            reloaded.eval()
            x = torch.stack([train_ds[i]["noisy"] for i in range(2)]).to(args.device)
            with torch.no_grad():
                out_original = model(x)
                out_reloaded = reloaded(x)
            model.train()
            assert torch.allclose(out_original, out_reloaded, atol=1e-5), \
                "reloaded checkpoint does not reproduce identical outputs"

    check("8_9_checkpoint_save_reload_identical_outputs", _checkpoint_roundtrip)

    # 10. evaluation on a few samples
    def _mini_eval():
        model.eval()
        val_ds = PrismNetTrainDataset(args.data_root, stems=val_stems[:5])
        with torch.no_grad():
            for i in range(len(val_ds)):
                sample = val_ds[i]
                pred = model(sample["noisy"].unsqueeze(0).to(args.device))
                p = compute_psnr(pred.squeeze(0).cpu(), sample["gt"])
                s = compute_ssim(pred.squeeze(0).cpu(), sample["gt"])
                assert p == p and s == s  # not NaN
        model.train()

    check("10_mini_evaluation", _mini_eval)

    print("\nALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()