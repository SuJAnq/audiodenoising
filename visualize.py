# visualization_colab.py
#
# INSTRUCTIONS FOR COLAB:
# 1. Ensure your project files (config.py, dataset.py, train.py, utilis.py, etc.) are in the current directory.
# 2. Install dependencies by running this in a cell:
#    !pip install pystoi matplotlib librosa
# 3. Run this script.

import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pystoi import stoi
from torch.utils.data import DataLoader

# Import project modules
# Ensure these files are uploaded to Colab
from config import get_config
from dataset import DenoisingDataset, DynamicDenoisingDataset
from model import UNet
from utilis import reconstruct_waveform_from_mag_and_phase, reconstruct_waveform_auto

cfg = get_config()

def get_loader(root, split="test", return_phase=True, dynamic=False):
    if dynamic:
        # For dynamic evaluation, we load raw audio files directly
        clean_files = sorted([os.path.join(cfg.CLEAN_EN_DIR, f) for f in os.listdir(cfg.CLEAN_EN_DIR) if f.endswith(".wav")])
        noise_files = []
        if os.path.exists(cfg.NOISE_DIR):
            for root_dir, dirs, files in os.walk(cfg.NOISE_DIR):
                for f in files:
                    if f.endswith(".wav"):
                        noise_files.append(os.path.join(root_dir, f))
        dataset = DynamicDenoisingDataset(clean_files, noise_files, split=split, return_phase=return_phase, cfg=cfg)
    else:
        dataset = DenoisingDataset(root, split=split, target="clean", return_phase=return_phase, cfg=cfg)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader, dataset

def plot_sample(model, batch, device, plot_dir, idx):
    noisy_log_mag, clean_log_mag, noisy_phase, clean_phase = batch

    name = f"sample_{idx}"
    
    # --- Inference ---
    noisy_log_mag = noisy_log_mag.to(device)
    with torch.no_grad():
        pred_log_mag = model(noisy_log_mag)

    # --- Reconstruction ---
    # Reconstruct audio (CPU)
    noisy_wav = reconstruct_waveform_from_mag_and_phase(noisy_log_mag.cpu(), noisy_phase)
    pred_wav = reconstruct_waveform_auto(
        pred_log_mag.cpu(),
        noisy_phase,
        ref_mag=noisy_log_mag.cpu(),
        prefer_input_phase=cfg.PREFER_INPUT_PHASE_RECON,
        apply_postfilter=cfg.APPLY_POSTFILTER,
    )
    clean_wav = reconstruct_waveform_from_mag_and_phase(clean_log_mag, clean_phase)


    # Convert to numpy for plotting
    noisy_spec = noisy_log_mag.squeeze().cpu().numpy()
    pred_spec = pred_log_mag.squeeze().cpu().numpy()
    clean_spec = clean_log_mag.squeeze().cpu().numpy()

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # helper to show spec
    def show_spec(ax, spec, title):
        img = librosa.display.specshow(spec, sr=cfg.SAMPLE_RATE, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(title)
        return img

    show_spec(axes[0], noisy_spec, "Noisy Input (Log Mag)")
    show_spec(axes[1], pred_spec, "Denoised Output (Log Mag)")
    show_spec(axes[2], clean_spec, "Clean Target (Log Mag)")
    
    plt.colorbar(axes[2].collections[0], ax=axes, orientation='horizontal', fraction=0.05)
    plt.suptitle(f"Sample {idx}")
    # plt.tight_layout()
    
    save_path = os.path.join(plot_dir, f"{name}_spectrogram.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

    # --- Save Audio ---
    audio_dir = os.path.join(cfg.RESULTS_DIR, "audio_samples")
    os.makedirs(audio_dir, exist_ok=True)
    
    # helper to ensure 1D array
    def to_numpy_audio(wav_tensor):
        return wav_tensor.squeeze().cpu().numpy()

    sf.write(os.path.join(audio_dir, f"{name}_noisy.wav"), to_numpy_audio(noisy_wav), cfg.SAMPLE_RATE)
    sf.write(os.path.join(audio_dir, f"{name}_denoised.wav"), to_numpy_audio(pred_wav), cfg.SAMPLE_RATE)
    sf.write(os.path.join(audio_dir, f"{name}_clean.wav"), to_numpy_audio(clean_wav), cfg.SAMPLE_RATE)
    print(f"Saved audio to {audio_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(cfg.CHECKPOINT_DIR, "unet_best.pt"), help="Path to checkpoint")
    parser.add_argument("--data-root", default=cfg.PROCESSED_ROOT, help="Path to data root")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic dataset generation")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    data_root = args.data_root
    plot_dir = os.path.join(cfg.RESULTS_DIR, "plots")
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load Model ---
    print(f"Loading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return

    model = UNet(in_ch=cfg.UNET_IN_CH, out_ch=cfg.UNET_OUT_CH, base_ch=cfg.UNET_BASE_CH).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print("✅ Model loaded successfully.")

    # --- Load Test Data ---
    loader, dataset = get_loader(data_root, split="test", return_phase=True, dynamic=args.dynamic)
    print(f"Test dataset size: {len(dataset)}")

    # --- Visualization ---
    print("\n--- Visualizing Samples ---")
    
    # Create directory for saving plots
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to {plot_dir} ...")

    # Visualize first N samples
    for idx, batch in enumerate(loader):
        if idx >= args.num_samples:
            break
        plot_sample(model, batch, DEVICE, plot_dir, idx)

if __name__ == "__main__":
    main()






