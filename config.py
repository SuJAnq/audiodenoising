"""Centralized training/evaluation configuration.

All runtime code should *read* configuration values through the frozen
``Config`` object returned by ``get_config``. To change a value, edit this
file or supply a temporary override via CLI args and ``get_config``.
Runtime mutation of config attributes is blocked to avoid accidental drift.
"""

import sys
import torch
import types
from typing import Any, Dict, Optional
from dataclasses import dataclass, replace

# Raw Data Paths
CLEAN_EN_DIR = "dataset/raw/clean/clean_en"
CLEAN_EN_MP3_DIR = "dataset/raw/clean/clean_en_mp3"
CLEAN_NP_DIR = "dataset/raw/clean/clean_np"
CLEAN_NP_WEBM_DIR = "dataset/raw/clean/clean_np_webm"
NOISE_DIR = "dataset/raw/noise"

# Root directory for processed spectrogram data
PROCESSED_ROOT = "dataset/processed"
PROCESSED_DIR = PROCESSED_ROOT  # Alias for compatibility
PROCESSED_AUDIO_DIR = "dataset/processed_audio"

# Directory to store checkpoints (Colab/Drive-friendly)
CHECKPOINT_DIR = "checkpoints"

# Directory to store evaluation results (Colab/Drive-friendly)
RESULTS_DIR = "results"

#Save visualized 

TRAIN_NOISY_DIR = f"{PROCESSED_ROOT}/train/noisy"
TRAIN_CLEAN_DIR = f"{PROCESSED_ROOT}/train/clean"

VAL_NOISY_DIR = f"{PROCESSED_ROOT}/val/noisy"
VAL_CLEAN_DIR = f"{PROCESSED_ROOT}/val/clean"

# Audio parameters

SAMPLE_RATE = 16000

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

FREQ_BINS = (N_FFT // 2) + 1   # 513 (raw STFT)
FREQ_BINS_MODEL = FREQ_BINS    # Model uses full 513-bin spectra

WINDOW_TYPE = "hann"
WINDOW = torch.hann_window(WIN_LENGTH)

# Preprocessing Config

NOISY_PAIRS_PER_CLEAN = 3             # Number of (noisy A, noisy B) pairs per clean clip
SNR_LIST = [-5, 0, 2.5, 5, 7.5, 10, 15]  # SNRs (dB) to sample from


# Spectrogram parameters

FIXED_TIME_FRAMES = 256        # must match preprocessing
USE_LOG_MAG = True


# Training parameters

BATCH_SIZE = 4  # Reduced for RTX 4050 6GB (UNet is memory-intensive)
ACCUMULATION_STEPS = 8  # Effective batch size = 4 * 8 = 32
GRADIENT_ACCUMULATION_STEPS = ACCUMULATION_STEPS
NUM_WORKERS = 0  # Set to 0 to avoid hanging in Colab/Jupyter parallel processing
PIN_MEMORY = False  # Pinning can add overhead on Colab's virtualized path
NOISE2NOISE = True  # When True, training targets are second noisy variants (Noise2Noise)
NUM_EPOCHS = 5

# Model architecture parameters
UNET_IN_CH = 1
UNET_OUT_CH = 1
UNET_BASE_CH = 64

# Optimizer and Scheduler parameters
INITIAL_LR = 2e-4
ADAM_BETAS = (0.9, 0.99)
USE_WEIGHT_DECAY = True
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
USE_SCHEDULER = True  # Flag to enable/disable scheduler


# Multi-Resolution STFT Loss parameters
LOSS_FUNCTION = "mse"  # Options: "mse", "l1", "mrstft", "hybrid", "hybrid_l1", "combined", "crm"
HYBRID_ALPHA = 0.5  # 0.5 means equal weight. < 0.5 favors MSE, > 0.5 favors MR-STFT
COMBINED_LAMBDA1 = 100.0  # Weight for waveform MSE Loss
COMBINED_LAMBDA2 = 1.0    # Weight for Multi-Resolution STFT Loss
CRM_MASK_ACTIVATION = "tanh"  # Activation for CRM mask components (recommended: tanh)

MR_STFT_FFT_SIZES = [1024, 2048, 512]
MR_STFT_HOP_SIZES = [120, 240, 50]
MR_STFT_WIN_LENGTHS = [600, 1200, 240]
MR_STFT_WINDOW = "hann_window"

# Griffin-Lim phase reconstruction parameters
GRIFFIN_LIM_ITER = 68          # Number of Griffin-Lim iterations

# Reconstruction policy and anti-hiss postfilter
PREFER_INPUT_PHASE_RECON = True   # If phase is available, prefer noisy-phase ISTFT over Griffin-Lim
APPLY_POSTFILTER = True           # Apply light postfilter to reduce static/hiss after reconstruction
POSTFILTER_CUTOFF_HZ = 6500.0     # Low-pass cutoff (Hz) for anti-hiss postfilter (was 7000)
POSTFILTER_STRENGTH = 0.35        # 0.0=no effect, 1.0=fully filtered (was 0.2)

# Spectral gating post-processor (attacks "rain" artifacts directly)
APPLY_SPECTRAL_GATE = True        # Zero-out quiet T-F bins below noise floor
SPECTRAL_GATE_THRESHOLD = 1.5     # Multiplier on estimated noise floor (higher = more aggressive)

# Wiener post-filter (soft SNR-based suppression)
APPLY_WIENER_POSTFILTER = True    # Apply Wiener-style post-filter after spectral gate
WIENER_BETA = 0.02                # Noise power estimate for Wiener; higher = more suppression

# Mask-based prediction mode (recommended for Stage 4+ training)
MASK_MODE = False                 # When True, UNet predicts a [0,1] ratio mask instead of magnitude


# Augmentation parameters

MIN_GAIN_DB = -10
MAX_GAIN_DB = 10
FREQ_MASK_PARAM = 20
TIME_MASK_PARAM = 30
NUM_FREQ_MASKS = 1
NUM_TIME_MASKS = 1


# Tools
FFMPEG_PATH = "ffmpeg"  # Assumes ffmpeg is in PATH (common for Linux)


# Visualization Defaults
VISUALIZATION_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/unet_best.pt"


# -----------------------------------------------------------------------------
# Frozen config object and safe accessors
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
	SAMPLE_RATE: int
	N_FFT: int
	HOP_LENGTH: int
	WIN_LENGTH: int
	FREQ_BINS: int
	FREQ_BINS_MODEL: int
	WINDOW_TYPE: str

	NOISY_PAIRS_PER_CLEAN: int
	SNR_LIST: tuple

	FIXED_TIME_FRAMES: int
	USE_LOG_MAG: bool

	BATCH_SIZE: int
	ACCUMULATION_STEPS: int
	GRADIENT_ACCUMULATION_STEPS: int
	NUM_WORKERS: int
	PIN_MEMORY: bool
	NOISE2NOISE: bool
	NUM_EPOCHS: int

	UNET_IN_CH: int
	UNET_OUT_CH: int
	UNET_BASE_CH: int

	INITIAL_LR: float
	ADAM_BETAS: tuple
	USE_WEIGHT_DECAY: bool
	WEIGHT_DECAY: float
	SCHEDULER_PATIENCE: int
	SCHEDULER_FACTOR: float
	USE_SCHEDULER: bool

	LOSS_FUNCTION: str
	HYBRID_ALPHA: float
	COMBINED_LAMBDA1: float
	COMBINED_LAMBDA2: float
	CRM_MASK_ACTIVATION: str
	MR_STFT_FFT_SIZES: tuple
	MR_STFT_HOP_SIZES: tuple
	MR_STFT_WIN_LENGTHS: tuple
	MR_STFT_WINDOW: str

	GRIFFIN_LIM_ITER: int
	PREFER_INPUT_PHASE_RECON: bool
	APPLY_POSTFILTER: bool
	POSTFILTER_CUTOFF_HZ: float
	POSTFILTER_STRENGTH: float

	APPLY_SPECTRAL_GATE: bool
	SPECTRAL_GATE_THRESHOLD: float
	APPLY_WIENER_POSTFILTER: bool
	WIENER_BETA: float
	MASK_MODE: bool

	PROCESSED_ROOT: str
	PROCESSED_DIR: str
	PROCESSED_AUDIO_DIR: str
	CHECKPOINT_DIR: str
	RESULTS_DIR: str
	TRAIN_NOISY_DIR: str
	TRAIN_CLEAN_DIR: str
	VAL_NOISY_DIR: str
	VAL_CLEAN_DIR: str

	MIN_GAIN_DB: float
	MAX_GAIN_DB: float
	FREQ_MASK_PARAM: int
	TIME_MASK_PARAM: int
	NUM_FREQ_MASKS: int
	NUM_TIME_MASKS: int

	CLEAN_EN_DIR: str
	CLEAN_EN_MP3_DIR: str
	CLEAN_NP_DIR: str
	CLEAN_NP_WEBM_DIR: str
	NOISE_DIR: str

	FFMPEG_PATH: str

	VISUALIZATION_CHECKPOINT_PATH: str


CONFIG = Config(
	SAMPLE_RATE=SAMPLE_RATE,
	N_FFT=N_FFT,
	HOP_LENGTH=HOP_LENGTH,
	WIN_LENGTH=WIN_LENGTH,
	FREQ_BINS=FREQ_BINS,
	FREQ_BINS_MODEL=FREQ_BINS_MODEL,
	WINDOW_TYPE=WINDOW_TYPE,
	NOISY_PAIRS_PER_CLEAN=NOISY_PAIRS_PER_CLEAN,
	SNR_LIST=tuple(SNR_LIST),
	FIXED_TIME_FRAMES=FIXED_TIME_FRAMES,
	USE_LOG_MAG=USE_LOG_MAG,
	BATCH_SIZE=BATCH_SIZE,
	ACCUMULATION_STEPS=ACCUMULATION_STEPS,
	GRADIENT_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS,
	NUM_WORKERS=NUM_WORKERS,
	PIN_MEMORY=PIN_MEMORY,
	NOISE2NOISE=NOISE2NOISE,
	NUM_EPOCHS=NUM_EPOCHS,
	UNET_IN_CH=UNET_IN_CH,
	UNET_OUT_CH=UNET_OUT_CH,
	UNET_BASE_CH=UNET_BASE_CH,
	INITIAL_LR=INITIAL_LR,
	ADAM_BETAS=tuple(ADAM_BETAS),
	USE_WEIGHT_DECAY=USE_WEIGHT_DECAY,
	WEIGHT_DECAY=WEIGHT_DECAY,
	SCHEDULER_PATIENCE=SCHEDULER_PATIENCE,
	SCHEDULER_FACTOR=SCHEDULER_FACTOR,
	USE_SCHEDULER=USE_SCHEDULER,
	LOSS_FUNCTION=LOSS_FUNCTION,
	HYBRID_ALPHA=HYBRID_ALPHA,
	COMBINED_LAMBDA1=COMBINED_LAMBDA1,
	COMBINED_LAMBDA2=COMBINED_LAMBDA2,
	CRM_MASK_ACTIVATION=CRM_MASK_ACTIVATION,
	MR_STFT_FFT_SIZES=tuple(MR_STFT_FFT_SIZES),
	MR_STFT_HOP_SIZES=tuple(MR_STFT_HOP_SIZES),
	MR_STFT_WIN_LENGTHS=tuple(MR_STFT_WIN_LENGTHS),
	MR_STFT_WINDOW=MR_STFT_WINDOW,
	GRIFFIN_LIM_ITER=GRIFFIN_LIM_ITER,
	PREFER_INPUT_PHASE_RECON=PREFER_INPUT_PHASE_RECON,
	APPLY_POSTFILTER=APPLY_POSTFILTER,
	POSTFILTER_CUTOFF_HZ=POSTFILTER_CUTOFF_HZ,
	POSTFILTER_STRENGTH=POSTFILTER_STRENGTH,
	APPLY_SPECTRAL_GATE=APPLY_SPECTRAL_GATE,
	SPECTRAL_GATE_THRESHOLD=SPECTRAL_GATE_THRESHOLD,
	APPLY_WIENER_POSTFILTER=APPLY_WIENER_POSTFILTER,
	WIENER_BETA=WIENER_BETA,
	MASK_MODE=MASK_MODE,
	PROCESSED_ROOT=PROCESSED_ROOT,
	PROCESSED_DIR=PROCESSED_DIR,
	PROCESSED_AUDIO_DIR=PROCESSED_AUDIO_DIR,
	CHECKPOINT_DIR=CHECKPOINT_DIR,
	RESULTS_DIR=RESULTS_DIR,
	TRAIN_NOISY_DIR=TRAIN_NOISY_DIR,
	TRAIN_CLEAN_DIR=TRAIN_CLEAN_DIR,
	VAL_NOISY_DIR=VAL_NOISY_DIR,
	VAL_CLEAN_DIR=VAL_CLEAN_DIR,
	MIN_GAIN_DB=MIN_GAIN_DB,
	MAX_GAIN_DB=MAX_GAIN_DB,
	FREQ_MASK_PARAM=FREQ_MASK_PARAM,
	TIME_MASK_PARAM=TIME_MASK_PARAM,
	NUM_FREQ_MASKS=NUM_FREQ_MASKS,
	NUM_TIME_MASKS=NUM_TIME_MASKS,
	CLEAN_EN_DIR=CLEAN_EN_DIR,
	CLEAN_EN_MP3_DIR=CLEAN_EN_MP3_DIR,
	CLEAN_NP_DIR=CLEAN_NP_DIR,
	CLEAN_NP_WEBM_DIR=CLEAN_NP_WEBM_DIR,
	NOISE_DIR=NOISE_DIR,
	FFMPEG_PATH=FFMPEG_PATH,
	VISUALIZATION_CHECKPOINT_PATH=VISUALIZATION_CHECKPOINT_PATH,
)


def get_config(overrides: Optional[Dict[str, Any]] = None) -> Config:
	"""Return a frozen Config, optionally with per-call overrides.

	Overrides do *not* mutate the global config; they return a copy suitable for
	a single run (e.g., CLI-specified hyperparameters).
	"""
	if not overrides:
		return CONFIG

	unknown = set(overrides) - set(Config.__annotations__)
	if unknown:
		raise KeyError(f"Unknown config override(s): {sorted(unknown)}")

	# Convert mutable lists to tuples to preserve immutability guarantees
	sanitized = {}
	for key, value in overrides.items():
		if isinstance(value, list):
			sanitized[key] = tuple(value)
		else:
			sanitized[key] = value

	return replace(CONFIG, **sanitized)


def _freeze_module_namespace():
	"""Prevent runtime mutation of config attributes via attribute assignment."""
	module = sys.modules[__name__]
	frozen_keys = set(module.__dict__.keys())

	class _FrozenConfig(types.ModuleType):
		def __setattr__(self, name: str, value: Any) -> None:
			if name.startswith("__") or name.startswith("_"):
				return super().__setattr__(name, value)
			if name in {"get_config", "CONFIG", "Config"}:
				raise AttributeError("Config is read-only at runtime; edit config.py or use get_config(overrides=...)")
			if name in frozen_keys:
				raise AttributeError("Config values cannot be reassigned at runtime; use get_config(overrides=...) instead")
			raise AttributeError("Cannot add new config entries at runtime; edit config.py")

	module.__class__ = _FrozenConfig
	module._frozen_keys = frozen_keys


_freeze_module_namespace()

