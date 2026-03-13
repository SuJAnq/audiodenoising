import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SAMPLE_RATE, PROCESSED_ROOT, NOISE2NOISE, RESULTS_DIR, CHECKPOINT_DIR,
    UNET_IN_CH, UNET_OUT_CH, UNET_BASE_CH, CLEAN_EN_DIR, NOISE_DIR,
    PREFER_INPUT_PHASE_RECON, APPLY_POSTFILTER
)
from dataset import DenoisingDataset, DynamicDenoisingDataset
from utilis import reconstruct_waveform_from_mag_and_phase, reconstruct_waveform_auto
from model import UNet

try:
    import soundfile as sf

    _HAS_SF = True
except Exception:
    _HAS_SF = False

try:
    from pystoi import stoi

    _HAS_STOI = True
except Exception:
    _HAS_STOI = False


def psnr_per_sample(clean: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute PSNR per sample. Assumes tensors of shape (B, C, H, W).
    Uses per-sample max of absolute clean signal as peak reference.
    """
    b = clean.size(0)
    clean_flat = clean.view(b, -1)
    pred_flat = pred.view(b, -1)
    mse = torch.mean((clean_flat - pred_flat) ** 2, dim=1)
    max_val = clean_flat.abs().amax(dim=1)
    # avoid division by zero
    mse = torch.clamp(mse, min=1e-12)
    max_val = torch.clamp(max_val, min=1e-12)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return psnr


def snr_per_sample(clean: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    b = clean.size(0)
    clean_flat = clean.view(b, -1)
    pred_flat = pred.view(b, -1)
    signal_power = torch.sum(clean_flat ** 2, dim=1)
    noise_power = torch.sum((clean_flat - pred_flat) ** 2, dim=1)
    
    # Avoid log(0) for extremely low signal power (silence)
    signal_power = torch.clamp(signal_power, min=1e-12)
    noise_power = torch.clamp(noise_power, min=1e-12)
    
    snr = 10.0 * torch.log10(signal_power / noise_power)
    return snr


def mse_l1_per_sample(clean: torch.Tensor, pred: torch.Tensor):
    b = clean.size(0)
    clean_flat = clean.view(b, -1)
    pred_flat = pred.view(b, -1)
    mse = torch.mean((clean_flat - pred_flat) ** 2, dim=1)
    l1 = torch.mean(torch.abs(clean_flat - pred_flat), dim=1)
    return mse, l1


def lsd_per_sample(clean: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log-spectral distance over frequency, averaged over time."""

    clean_mag = clean.squeeze(1)
    pred_mag = pred.squeeze(1)
    clean_log = torch.log10(clean_mag + eps)
    pred_log = torch.log10(pred_mag + eps)
    diff2 = (clean_log - pred_log) ** 2
    # mean over freq then sqrt per frame, then mean over time -> per sample
    per_frame = torch.sqrt(diff2.mean(dim=1))
    return per_frame.mean(dim=1)


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim_per_sample(clean: torch.Tensor, pred: torch.Tensor, window: torch.Tensor = None, window_size: int = 11) -> torch.Tensor:
    """Compute SSIM per sample for spectrogram-like tensors (B,1,F,T)."""

    if window is None or window.device != clean.device or window.dtype != clean.dtype:
        window = _gaussian_window(window_size=window_size).to(device=clean.device, dtype=clean.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(clean, window, padding=pad, groups=1)
    mu_y = F.conv2d(pred, window, padding=pad, groups=1)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(clean * clean, window, padding=pad, groups=1) - mu_x2
    sigma_y2 = F.conv2d(pred * pred, window, padding=pad, groups=1) - mu_y2
    sigma_xy = F.conv2d(clean * pred, window, padding=pad, groups=1) - mu_xy

    data_range = (torch.max(torch.stack([clean.max(), pred.max()])) - torch.min(torch.stack([clean.min(), pred.min()]))).clamp(min=1e-4)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    b = clean.size(0)
    return ssim_map.view(b, -1).mean(dim=1)


def get_val_loader(root: str, split: str = "val", return_phase: bool = False, target: str = "clean", dynamic: bool = False):
    if dynamic:
        # For dynamic evaluation, we load raw audio files directly
        # Note: clean_en is just one example; you might want to evaluate on clean_np, etc.
        # or combine them. Here, we stick to CLEAN_EN_DIR for consistency with training demo.
        clean_files = sorted([os.path.join(CLEAN_EN_DIR, f) for f in os.listdir(CLEAN_EN_DIR) if f.endswith(".wav")])
        
        # Collect all noise files
        noise_files = []
        if os.path.exists(NOISE_DIR):
             for root, dirs, files in os.walk(NOISE_DIR):
                for f in files:
                    if f.endswith(".wav"):
                        noise_files.append(os.path.join(root, f))
        
        val_ds = DynamicDenoisingDataset(clean_files, noise_files, split=split, return_phase=return_phase, target=target)
    else:
        val_ds = DenoisingDataset(root, split=split, return_phase=return_phase, target=target)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return val_loader


@torch.no_grad()
def evaluate(
    checkpoint: str,
    data_root: str = PROCESSED_ROOT,
    split: str = "val",
    reconstruct: bool = False,
    out_dir: str = f"{RESULTS_DIR}/recon",
    enable_stoi: bool = True,
    compare_to_clean: bool = True,
    dynamic: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if CUDA is actually usable despite being "available"
    if device.type == "cuda":
        try:
            # Check capabilities
            cap = torch.cuda.get_device_capability(device)
            if cap[0] < 7 or (cap[0] == 7 and cap[1] < 5):
                 print(f"WARNING: Detected GPU with capability {cap[0]}.{cap[1]}. This PyTorch version may require 7.5+.")
                 print("Examples of 6.1 GPUs (unsupported by this PT binary): GTX 1050/1060/1070/1080.")
                 print("Continuing with CUDA enabled (may crash if unsupported ops are used).")
        except Exception:
            pass

    print(f"Using device: {device}")

    model = UNet(in_ch=UNET_IN_CH, out_ch=UNET_OUT_CH, base_ch=UNET_BASE_CH).to(device)
    state = torch.load(checkpoint, map_location=device)
    # Support both raw state_dict (old behavior) and full checkpoint dict
    state_dict = state.get("model_state_dict") if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.eval()

    # If clean targets are available, keep compare_to_clean=True (recommended for metrics).
    # If evaluating strictly Noise2Noise targets, set compare_to_clean=False to compare against noisy-B.
    target_mode = "clean" if compare_to_clean else "noisy"
    need_phase = reconstruct or (enable_stoi and _HAS_STOI)
    val_loader = get_val_loader(data_root, split=split, return_phase=need_phase, target=target_mode, dynamic=dynamic)
    all_psnr = []
    all_snr = []
    all_mse = []
    all_l1 = []
    all_lsd = []
    all_ssim = []
    all_stoi = []
    file_counter = 0
    num_processed = 0
    window = None

    for batch in val_loader:
        if need_phase:
            # target_mode dictates what phases come back
            if target_mode == "clean":
                noisy, target_mag, noisy_phase, clean_phase = batch
            else:
                noisy, target_mag, noisy_phase, _ = batch  # second phase unused here
                clean_phase = None
        else:
            noisy, target_mag = batch
            noisy_phase = None
            clean_phase = None

        noisy = noisy.to(device)
        target_mag = target_mag.to(device)
        pred = model(noisy)
        pred = pred.type_as(target_mag)

        psnr_vals = psnr_per_sample(target_mag, pred)
        snr_vals = snr_per_sample(target_mag, pred)
        mse_vals, l1_vals = mse_l1_per_sample(target_mag, pred)
        lsd_vals = lsd_per_sample(target_mag, pred)
        if window is None or window.device != pred.device or window.dtype != pred.dtype:
            window = _gaussian_window().to(device=pred.device, dtype=pred.dtype)
        ssim_vals = ssim_per_sample(target_mag, pred, window=window)

        all_psnr.append(psnr_vals.cpu())
        all_snr.append(snr_vals.cpu())
        all_mse.append(mse_vals.cpu())
        all_l1.append(l1_vals.cpu())
        all_lsd.append(lsd_vals.cpu())
        all_ssim.append(ssim_vals.cpu())

        num_processed += noisy.size(0)
        if num_processed % 4 == 0:  # Print frequently since CPU is slow
             print(f"Processed {num_processed} samples...", end='\r', flush=True)

        if compare_to_clean and enable_stoi and _HAS_STOI and noisy_phase is not None and clean_phase is not None:
            pred_mag = pred.squeeze(1).detach().cpu()
            noisy_phase_cpu = noisy_phase.cpu()
            clean_mag = target_mag.squeeze(1).detach().cpu()
            clean_phase_cpu = clean_phase.cpu()

            try:
                # Pass noisy magnitude (input) as reference for Nyquist bin
                # The `noisy` variable holds the input magnitude
                pred_wav = reconstruct_waveform_from_mag_and_phase(
                    pred_mag, 
                    noisy_phase_cpu,
                    ref_mag=noisy.cpu()
                )
                clean_wav = reconstruct_waveform_from_mag_and_phase(clean_mag, clean_phase_cpu)

                stoi_scores = []
                for pw, cw in zip(pred_wav, clean_wav):
                    min_len = min(pw.numel(), cw.numel())
                    if min_len == 0:
                        stoi_scores.append(torch.tensor(0.0))
                        continue
                    score = stoi(
                        cw[:min_len].numpy(),
                        pw[:min_len].numpy(),
                        SAMPLE_RATE,
                        extended=False,
                    )
                    stoi_scores.append(torch.tensor(float(score)))

                if len(stoi_scores) > 0:
                    all_stoi.append(torch.stack(stoi_scores))
            except Exception:
                # If reconstruction fails, skip STOI for this batch
                pass

        if reconstruct:
            pred_mag = pred.squeeze(1).cpu()
            noisy_phase_cpu = noisy_phase.cpu() if noisy_phase is not None else None
            # Pass original noisy magnitude (target_mag actually usually is clean/target, we need noisy for reference)
            # But the loop loads noisy as (B, 1, F, T) -> noisy.
            noisy_mag_cpu = noisy.cpu()
            
            wave_batch = reconstruct_waveform_auto(
                pred_mag,
                noisy_phase_cpu,
                ref_mag=noisy_mag_cpu,
                prefer_input_phase=PREFER_INPUT_PHASE_RECON,
                apply_postfilter=APPLY_POSTFILTER,
            )
            os.makedirs(out_dir, exist_ok=True)
            for i, waveform in enumerate(wave_batch):
                path = os.path.join(out_dir, f"recon_{file_counter:08d}.wav")
                wav_np = waveform.detach().numpy()
                if _HAS_SF:
                    sf.write(path, wav_np, samplerate=SAMPLE_RATE)
                file_counter += 1

    print("") # Newline
    all_psnr = torch.cat(all_psnr)
    all_snr = torch.cat(all_snr)
    all_mse = torch.cat(all_mse)
    all_l1 = torch.cat(all_l1)
    all_lsd = torch.cat(all_lsd)
    all_ssim = torch.cat(all_ssim)
    all_stoi_cat = torch.cat(all_stoi) if len(all_stoi) > 0 else None

    print(f"Validation samples: {all_psnr.numel()}")
    print(f"Average PSNR: {all_psnr.mean().item():.3f} dB")
    print(f"Median PSNR: {all_psnr.median().item():.3f} dB")
    print(f"Average SNR: {all_snr.mean().item():.3f} dB")
    print(f"Median SNR: {all_snr.median().item():.3f} dB")
    print(f"Average MSE: {all_mse.mean().item():.6f}")
    print(f"Average L1: {all_l1.mean().item():.6f}")
    print(f"Average LSD: {all_lsd.mean().item():.6f}")
    print(f"Average SSIM: {all_ssim.mean().item():.6f}")
    if enable_stoi:
        if _HAS_STOI and all_stoi_cat is not None:
            print(f"Average STOI: {all_stoi_cat.mean().item():.6f}")
        elif not _HAS_STOI:
            print("STOI skipped: install `pystoi` to enable.")
        else:
            print("STOI skipped: phase or reconstruction unavailable.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate UNet denoiser on validation or test set")
    # Use config paths as defaults
    ckpt_default = os.path.join(CHECKPOINT_DIR, "unet_placeholder.pt")
    
    p.add_argument("--checkpoint", default=ckpt_default, help="Path to model checkpoint")
    p.add_argument("--data-root", default=PROCESSED_ROOT, help="Processed dataset root")
    p.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate (val or test)")
    p.add_argument("--reconstruct", action="store_true", help="Reconstruct waveforms using noisy phase and save to --out-dir")
    p.add_argument("--out-dir", default=f"{RESULTS_DIR}/recon", help="Output directory for reconstructed WAVs")
    p.add_argument("--no-stoi", action="store_true", help="Skip STOI computation even if pystoi is installed")
    p.add_argument("--dynamic", action="store_true", help="Use dynamic dataset generation (on-the-fly mixing)")
    
    # Use parse_known_args to handle Jupyter/Colab kernel arguments (like -f ...) gracefully
    args, _ = p.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        args.checkpoint,
        args.data_root,
        split=args.split,
        reconstruct=args.reconstruct,
        out_dir=args.out_dir,
        enable_stoi=not args.no_stoi,
        dynamic=args.dynamic,
    )