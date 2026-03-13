"""Compute baseline (noisy-input) metrics on validation set.
Usage: python tools/baseline_metrics.py
"""
import torch
from evaluate import get_val_loader, psnr_per_sample, snr_per_sample, mse_l1_per_sample, lsd_per_sample, ssim_per_sample, _gaussian_window
from config import PROCESSED_ROOT

loader = get_val_loader(PROCESSED_ROOT, split='val', return_phase=False, target='clean')
all_psnr = []
all_snr = []
all_mse = []
all_l1 = []
all_lsd = []
all_ssim = []
window = None

for noisy, target in loader:
    psnr_vals = psnr_per_sample(target, noisy)
    snr_vals = snr_per_sample(target, noisy)
    mse_vals, l1_vals = mse_l1_per_sample(target, noisy)
    lsd_vals = lsd_per_sample(target, noisy)
    if window is None or window.device != target.device or window.dtype != target.dtype:
        window = _gaussian_window().to(device=target.device, dtype=target.dtype)
    ssim_vals = ssim_per_sample(target, noisy, window=window)

    all_psnr.append(psnr_vals.cpu())
    all_snr.append(snr_vals.cpu())
    all_mse.append(mse_vals.cpu())
    all_l1.append(l1_vals.cpu())
    all_lsd.append(lsd_vals.cpu())
    all_ssim.append(ssim_vals.cpu())

all_psnr = torch.cat(all_psnr)
all_snr = torch.cat(all_snr)
all_mse = torch.cat(all_mse)
all_l1 = torch.cat(all_l1)
all_lsd = torch.cat(all_lsd)
all_ssim = torch.cat(all_ssim)

print(f"Baseline (noisy input) - Validation samples: {all_psnr.numel()}")
print(f"Avg PSNR: {all_psnr.mean().item():.3f} dB")
print(f"Med PSNR: {all_psnr.median().item():.3f} dB")
print(f"Avg SNR: {all_snr.mean().item():.3f} dB")
print(f"Med SNR: {all_snr.median().item():.3f} dB")
print(f"Avg MSE: {all_mse.mean().item():.6f}")
print(f"Avg L1: {all_l1.mean().item():.6f}")
print(f"Avg LSD: {all_lsd.mean().item():.6f}")
print(f"Avg SSIM: {all_ssim.mean().item():.6f}")
