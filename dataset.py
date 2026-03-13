# dataset.py
import os
import random
import warnings
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from config import get_config

# Helpers for dynamic mixing
def load_audio(path: str, sr: int) -> np.ndarray:
    """Load audio as 1D numpy array with the provided sample rate."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"PySoundFile failed\. Trying audioread instead\.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"librosa\.core\.audio\.__audioread_load",
            category=FutureWarning,
        )
        audio, _ = librosa.load(path, sr=sr)
    return audio


def fix_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim audio to a fixed length (loops if shorter)."""

    if len(audio) < target_len:
        n_repeats = int(np.ceil(target_len / len(audio)))
        audio = np.tile(audio, n_repeats)
    return audio[:target_len]


def add_noise(clean_audio: np.ndarray, noise_audio: np.ndarray, snr_db: float = 5) -> np.ndarray:
    """Mix noise with clean audio at the specified SNR."""

    noise_audio = noise_audio[:len(clean_audio)]
    clean_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_audio = noise_audio * np.sqrt(target_noise_power / (noise_power + 1e-8))
    return clean_audio + noise_audio


def compute_stft_tensor(
    audio: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute STFT magnitude and phase tensors using provided parameters."""

    waveform = torch.tensor(audio, dtype=torch.float32)
    if window is None:
        window = torch.hann_window(win_length)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    return mag, phase

class DenoisingDataset(Dataset):
    """
    Dataset for audio denoising using precomputed STFT tensors (.pt files).
    (Legacy support for static files)
    """
    def __init__(self, root_dir, split="train", return_phase: bool = False, target: str = "clean", cfg=None):
        """
        root_dir : str : path to processed folder
        split    : str : "train", "val", or "test"
        target   : "clean" for supervised denoising, "noisy" for Noise2Noise (input noisy A, target noisy B)
        cfg      : Config : optional immutable config to share overrides across components
        """
        assert split in ["train", "val", "test"], "split must be train/val/test"
        assert target in ["clean", "noisy"], "target must be 'clean' or 'noisy'"

        self.cfg = cfg or get_config()
        self.clean_dir = os.path.join(root_dir, split, "clean")
        self.noisy_dir = os.path.join(root_dir, split, "noisy")
        self.split = split
        self.return_phase = return_phase
        self.target = target

        # Cached config values for readability
        self.fixed_frames = self.cfg.FIXED_TIME_FRAMES
        self.use_log_mag = self.cfg.USE_LOG_MAG
        self.freq_bins = self.cfg.FREQ_BINS

        # List all files and sort to ensure clean/noisy pairing
        self.clean_files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith(".pt")])
        self.noisy_files = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith(".pt")])

        # Sanity check
        assert len(self.clean_files) == len(self.noisy_files), \
            f"Mismatch: {len(self.clean_files)} clean vs {len(self.noisy_files)} noisy files"

        self.length = len(self.clean_files)

    def __len__(self):
        return self.length

    def random_gain(self, mag):
        gain_db = torch.empty(1).uniform_(self.cfg.MIN_GAIN_DB, self.cfg.MAX_GAIN_DB).item()
        gain = 10 ** (gain_db / 20)
        return mag * gain

    def spec_augment(self, mag):
        augmented_mag = mag.clone()
        freq_bins, time_steps = augmented_mag.shape
        
        # Frequency Masking
        for _ in range(self.cfg.NUM_FREQ_MASKS):
            f = int(torch.empty(1).uniform_(0, self.cfg.FREQ_MASK_PARAM).item())
            if f > 0:
                f0 = int(torch.empty(1).uniform_(0, freq_bins - f).item())
                augmented_mag[f0:f0 + f, :] = 0.0
                
        # Time Masking
        for _ in range(self.cfg.NUM_TIME_MASKS):
            t = int(torch.empty(1).uniform_(0, self.cfg.TIME_MASK_PARAM).item())
            if t > 0:
                t0 = int(torch.empty(1).uniform_(0, time_steps - t).item())
                augmented_mag[:, t0:t0 + t] = 0.0
                
        return augmented_mag

    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])

        loaded_clean = torch.load(clean_path, weights_only=False)
        loaded_noisy = torch.load(noisy_path, weights_only=False)

        # Support legacy single-tensor saves (mag only) and new (mag, phase) tuples
        if isinstance(loaded_clean, torch.Tensor):
            clean = loaded_clean.float()
            clean_phase = None
        else:
            # expect (mag, phase) or {'mag':..., 'phase':...}
            if isinstance(loaded_clean, dict):
                clean = loaded_clean['mag'].float()
                clean_phase = loaded_clean.get('phase', None)
            else:
                clean, clean_phase = loaded_clean
                clean = clean.float()

        noisy_phase_a = None
        noisy_phase_b = None

        # Allow saved formats:
        # - Tensor (single noisy) -> duplicated for Noise2Noise target
        # - Tuple (mag, phase)
        # - Tuple (noisyA_mag, noisyA_phase, noisyB_mag, noisyB_phase)
        # - Dict with keys a_mag, b_mag, a_phase, b_phase
        if isinstance(loaded_noisy, torch.Tensor):
            noisy_a = loaded_noisy.float()
            noisy_b = loaded_noisy.float()
        elif isinstance(loaded_noisy, dict):
            if "a_mag" in loaded_noisy and "b_mag" in loaded_noisy:
                noisy_a = loaded_noisy["a_mag"].float()
                noisy_b = loaded_noisy["b_mag"].float()
                noisy_phase_a = loaded_noisy.get("a_phase", None)
                noisy_phase_b = loaded_noisy.get("b_phase", None)
            else:
                noisy = loaded_noisy['mag'].float()
                noisy_phase_a = loaded_noisy.get('phase', None)
                noisy_a = noisy
                noisy_b = noisy
                noisy_phase_b = noisy_phase_a
        else:
            if len(loaded_noisy) == 2:
                noisy, noisy_phase = loaded_noisy
                noisy = noisy.float()
                noisy_a = noisy
                noisy_b = noisy
                noisy_phase_a = noisy_phase
                noisy_phase_b = noisy_phase
            elif len(loaded_noisy) == 4:
                noisy_a, noisy_phase_a, noisy_b, noisy_phase_b = loaded_noisy
                noisy_a = noisy_a.float()
                noisy_b = noisy_b.float()
            else:
                raise ValueError("Unsupported noisy tensor format")

        # Safety checks
        assert clean.shape[0] == self.freq_bins, f"Clean tensor has {clean.shape[0]} freq bins, expected {self.freq_bins}"
        assert noisy_a.shape[0] == self.freq_bins, f"Noisy tensor A has {noisy_a.shape[0]} freq bins, expected {self.freq_bins}"
        assert noisy_b.shape[0] == self.freq_bins, f"Noisy tensor B has {noisy_b.shape[0]} freq bins, expected {self.freq_bins}"

        # Pad or crop time dimension to fixed size
        if self.fixed_frames is not None:
            def _pad_crop(spec, phase):
                if spec.shape[1] < self.fixed_frames:
                    spec = torch.nn.functional.pad(spec, (0, self.fixed_frames - spec.shape[1]))
                    if phase is not None:
                        phase = torch.nn.functional.pad(phase, (0, self.fixed_frames - phase.shape[1]))
                else:
                    spec = spec[:, :self.fixed_frames]
                    if phase is not None:
                        phase = phase[:, :self.fixed_frames]
                return spec, phase

            clean, clean_phase = _pad_crop(clean, clean_phase)
            noisy_a, noisy_phase_a = _pad_crop(noisy_a, noisy_phase_a)
            noisy_b, noisy_phase_b = _pad_crop(noisy_b, noisy_phase_b)

        # Apply Augmentation (Only strictly if split is training)
        if self.split == 'train':
            noisy_a = self.random_gain(noisy_a)
            noisy_a = self.spec_augment(noisy_a)

        # Optional log-magnitude (note: phases are stored as angles; leave unchanged)
        if self.use_log_mag:
            clean = torch.log1p(clean)
            noisy_a = torch.log1p(noisy_a)
            noisy_b = torch.log1p(noisy_b)

        # Add channel dimension for CNN
        clean = clean.unsqueeze(0)  # (1, freq_bins, T)
        noisy_a = noisy_a.unsqueeze(0)
        noisy_b = noisy_b.unsqueeze(0)

        # Select target for training/eval
        target_mag = clean if self.target == "clean" else noisy_b

        if self.return_phase:
            # return phase without channel dim so batch shape is (B, freq, T)
            def _phase_or_dummy(phase):
                if phase is None:
                    return torch.zeros(clean.shape[1], clean.shape[2])
                return phase

            noisy_phase_a = _phase_or_dummy(noisy_phase_a)
            noisy_phase_b = _phase_or_dummy(noisy_phase_b)
            clean_phase = _phase_or_dummy(clean_phase)

            return noisy_a, target_mag, noisy_phase_a, noisy_phase_b if self.target == "noisy" else clean_phase

        return noisy_a, target_mag
class DynamicDenoisingDataset(Dataset):
    """
    Dataset for on-the-fly audio denoising.
    Mixes clean speech and noise dynamically with random SNR.
    """
    def __init__(self, clean_files, noise_files, split="train", return_phase=False, target="clean", cfg=None):
        self.cfg = cfg or get_config()
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.split = split
        self.return_phase = return_phase
        self.target = target
        self.window = torch.hann_window(self.cfg.WIN_LENGTH)
        self.target_len = self.cfg.SAMPLE_RATE * 4

    def __len__(self):
        return len(self.clean_files)

    def random_gain(self, mag):
        gain_db = torch.empty(1).uniform_(self.cfg.MIN_GAIN_DB, self.cfg.MAX_GAIN_DB).item()
        gain = 10 ** (gain_db / 20)
        return mag * gain

    def spec_augment(self, mag):
        augmented_mag = mag.clone()
        freq_bins, time_steps = augmented_mag.shape
        for _ in range(self.cfg.NUM_FREQ_MASKS):
            f = int(torch.empty(1).uniform_(0, self.cfg.FREQ_MASK_PARAM).item())
            if f > 0:
                f0 = int(torch.empty(1).uniform_(0, freq_bins - f).item())
                augmented_mag[f0:f0 + f, :] = 0.0
        for _ in range(self.cfg.NUM_TIME_MASKS):
            t = int(torch.empty(1).uniform_(0, self.cfg.TIME_MASK_PARAM).item())
            if t > 0:
                t0 = int(torch.empty(1).uniform_(0, time_steps - t).item())
                augmented_mag[:, t0:t0 + t] = 0.0
        return augmented_mag

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        
        # Load Clean
        clean_audio = load_audio(clean_path, sr=self.cfg.SAMPLE_RATE)
        clean_audio = fix_length(clean_audio, target_len=self.target_len)

        # -------------------------------------------------------
        # Path 1: Noise2Noise (Target = Second Noisy Version)
        # -------------------------------------------------------
        if self.target == "noisy":
            # Generate Noisy A (Input)
            noise_path_a = random.choice(self.noise_files)
            noise_audio_a = fix_length(load_audio(noise_path_a, sr=self.cfg.SAMPLE_RATE), target_len=self.target_len)
            snr_a = random.choice(self.cfg.SNR_LIST) if self.split == "train" else 5.0
            noisy_audio_a = add_noise(clean_audio, noise_audio_a, snr_a)

            # Generate Noisy B (Target) - MUST be a different noise file for independence
            # Correlated noise between A and B causes residual static/rain artifacts
            remaining = [p for p in self.noise_files if p != noise_path_a]
            noise_path_b = random.choice(remaining) if remaining else random.choice(self.noise_files)
            noise_audio_b = fix_length(load_audio(noise_path_b, sr=self.cfg.SAMPLE_RATE), target_len=self.target_len)
            snr_b = random.choice(self.cfg.SNR_LIST) if self.split == "train" else 5.0
            noisy_audio_b = add_noise(clean_audio, noise_audio_b, snr_b)

            # Compute STFTs
            noisy_mag_a, noisy_phase_a = compute_stft_tensor(
                noisy_audio_a,
                n_fft=self.cfg.N_FFT,
                hop_length=self.cfg.HOP_LENGTH,
                win_length=self.cfg.WIN_LENGTH,
                window=self.window,
            )
            noisy_mag_b, noisy_phase_b = compute_stft_tensor(
                noisy_audio_b,
                n_fft=self.cfg.N_FFT,
                hop_length=self.cfg.HOP_LENGTH,
                win_length=self.cfg.WIN_LENGTH,
                window=self.window,
            )

            input_mag = noisy_mag_a
            target_mag = noisy_mag_b
            input_phase = noisy_phase_a
            target_phase = noisy_phase_b
        
        # -------------------------------------------------------
        # Path 2: Standard Supervised (Target = Clean)
        # -------------------------------------------------------
        else:
            noise_path = random.choice(self.noise_files)
            noise_audio = fix_length(load_audio(noise_path, sr=self.cfg.SAMPLE_RATE), target_len=self.target_len)
            snr = random.choice(self.cfg.SNR_LIST) if self.split == "train" else 5.0
            noisy_audio = add_noise(clean_audio, noise_audio, snr)

            input_mag, input_phase = compute_stft_tensor(
                noisy_audio,
                n_fft=self.cfg.N_FFT,
                hop_length=self.cfg.HOP_LENGTH,
                win_length=self.cfg.WIN_LENGTH,
                window=self.window,
            )
            clean_mag, clean_phase = compute_stft_tensor(
                clean_audio,
                n_fft=self.cfg.N_FFT,
                hop_length=self.cfg.HOP_LENGTH,
                win_length=self.cfg.WIN_LENGTH,
                window=self.window,
            )
            
            input_mag = input_mag
            target_mag = clean_mag
            target_phase = clean_phase

        # Common Post-Processing --------------------------------

        # Pad or crop time dimension
        if self.cfg.FIXED_TIME_FRAMES is not None:
            def _pad_crop_spec(spec, phase):
                if spec.shape[1] < self.cfg.FIXED_TIME_FRAMES:
                    spec = torch.nn.functional.pad(spec, (0, self.cfg.FIXED_TIME_FRAMES - spec.shape[1]))
                    if phase is not None:
                        phase = torch.nn.functional.pad(phase, (0, self.cfg.FIXED_TIME_FRAMES - phase.shape[1]))
                else:
                    spec = spec[:, :self.cfg.FIXED_TIME_FRAMES]
                    if phase is not None:
                        phase = phase[:, :self.cfg.FIXED_TIME_FRAMES]
                return spec, phase

            input_mag, input_phase = _pad_crop_spec(input_mag, input_phase)
            target_mag, target_phase = _pad_crop_spec(target_mag, target_phase)

        # Augment Input (Only Training)
        if self.split == "train":
            input_mag = self.random_gain(input_mag)
            input_mag = self.spec_augment(input_mag)

        # Log Transform
        if self.cfg.USE_LOG_MAG:
            input_mag = torch.log1p(input_mag)
            target_mag = torch.log1p(target_mag)

        # Add Channel Dim
        input_mag = input_mag.unsqueeze(0)
        target_mag = target_mag.unsqueeze(0)

        if self.return_phase:
            return input_mag, target_mag, input_phase, target_phase
        
        return input_mag, target_mag
