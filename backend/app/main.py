"""FastAPI backend for audio denoising.

Serves the U-Net model and provides:
  POST /api/denoise  — accepts audio file, returns denoised WAV
  GET  /api/health   — liveness / readiness check
"""

import os
import time
import uuid
import math
import tempfile
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

try:
    import soundfile as sf
except Exception:
    sf = None

# ── Resolve project root so we can import model / config ──
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # MinorProject/
sys.path.insert(0, str(PROJECT_ROOT))

from model import UNet
from config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH, USE_LOG_MAG, MASK_MODE, LOSS_FUNCTION,
)

# Import model architecture params with safe fallback
try:
    from config import UNET_IN_CH, UNET_OUT_CH, UNET_BASE_CH
except ImportError:
    UNET_IN_CH, UNET_OUT_CH, UNET_BASE_CH = 1, 1, 64

# Try to import WINDOW from config; fall back to generating it
try:
    from config import WINDOW as _CFG_WINDOW
except ImportError:
    _CFG_WINDOW = None

# Optional post-filter defaults from training config.
try:
    from config import (
        APPLY_POSTFILTER as _CFG_APPLY_POSTFILTER,
        POSTFILTER_CUTOFF_HZ as _CFG_POSTFILTER_CUTOFF_HZ,
        POSTFILTER_STRENGTH as _CFG_POSTFILTER_STRENGTH,
        APPLY_SPECTRAL_GATE as _CFG_APPLY_SPECTRAL_GATE,
        SPECTRAL_GATE_THRESHOLD as _CFG_SPECTRAL_GATE_THRESHOLD,
        APPLY_WIENER_POSTFILTER as _CFG_APPLY_WIENER_POSTFILTER,
        WIENER_BETA as _CFG_WIENER_BETA,
    )
except ImportError:
    _CFG_APPLY_POSTFILTER = True
    _CFG_POSTFILTER_CUTOFF_HZ = 6500.0
    _CFG_POSTFILTER_STRENGTH = 0.35
    _CFG_APPLY_SPECTRAL_GATE = True
    _CFG_SPECTRAL_GATE_THRESHOLD = 1.5
    _CFG_APPLY_WIENER_POSTFILTER = True
    _CFG_WIENER_BETA = 0.02

try:
    from config import GRIFFIN_LIM_ITER as _CFG_GRIFFIN_LIM_ITER
except ImportError:
    _CFG_GRIFFIN_LIM_ITER = 32

logger = logging.getLogger("denoiser")
logging.basicConfig(level=logging.INFO)

METRIC_HEADER_NAMES = [
    "X-Metric-Before-SNR",
    "X-Metric-Before-PSNR",
    "X-Metric-Before-SSIM",
    "X-Metric-Before-LSD",
    "X-Metric-After-SNR",
    "X-Metric-After-PSNR",
    "X-Metric-After-SSIM",
    "X-Metric-After-LSD",
]


def _parse_allowed_origins() -> list[str]:
    """Read CORS origins from ALLOWED_ORIGINS env var.

    ALLOWED_ORIGINS should be a comma-separated list, for example:
      https://my-frontend.vercel.app,http://localhost:3000
    """
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not raw:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using default %s.", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%s is below minimum %s. Using default %s.", name, value, minimum, default)
        return default
    return value


def _float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r. Using default %s.", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("%s=%s is below minimum %s. Using default %s.", name, value, minimum, default)
        return default
    if maximum is not None and value > maximum:
        logger.warning("%s=%s is above maximum %s. Using default %s.", name, value, maximum, default)
        return default
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid boolean for %s=%r. Using default %s.", name, raw, default)
    return default


def _pick_checkpoint_from_dir(directory: Path) -> Path | None:
    if not directory.exists() or not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.pt"))
    best = [c for c in candidates if "best" in c.stem]
    if best:
        return best[-1]
    if candidates:
        return candidates[-1]
    return None


# ── Globals ──
_model: UNet | None = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints"))).expanduser()
FALLBACK_CHECKPOINT_DIR = PROJECT_ROOT / "backend" / "models"
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
TEMP_DIR = Path(tempfile.gettempdir()) / "denoiser_outputs"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_ORIGINS = _parse_allowed_origins()
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS

# Limit thread fan-out in constrained runtimes (e.g., HF cpu-basic) to avoid
# memory spikes during startup and model execution.
TORCH_THREADS = _int_env("TORCH_THREADS", 1, minimum=1)
TORCH_INTEROP_THREADS = _int_env("TORCH_INTEROP_THREADS", 1, minimum=1)
try:
    torch.set_num_threads(TORCH_THREADS)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
except Exception as exc:
    logger.warning("Unable to configure PyTorch thread limits: %s", exc)

# ── Accepted audio extensions ──
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".webm"}
MAX_DURATION_SEC = _int_env("MAX_DURATION_SEC", 325, minimum=10)
UPLOAD_CHUNK_BYTES = _int_env("UPLOAD_CHUNK_BYTES", 1024 * 1024, minimum=64 * 1024)

# Long-audio safety guards to keep the API responsive on constrained CPUs.
LONG_AUDIO_SEC = _int_env("LONG_AUDIO_SEC", 150, minimum=30)
LONG_AUDIO_DISABLE_GRIFFIN_LIM = _bool_env("LONG_AUDIO_DISABLE_GRIFFIN_LIM", True)
LONG_AUDIO_GRIFFIN_LIM_ITER = _int_env("LONG_AUDIO_GRIFFIN_LIM_ITER", 12, minimum=1)

# Optional post-filter controls (env overrides config defaults).
APPLY_POSTFILTER = _bool_env("APPLY_POSTFILTER", _CFG_APPLY_POSTFILTER)
POSTFILTER_CUTOFF_HZ = _float_env("POSTFILTER_CUTOFF_HZ", _CFG_POSTFILTER_CUTOFF_HZ, minimum=1000.0)
POSTFILTER_STRENGTH = _float_env("POSTFILTER_STRENGTH", _CFG_POSTFILTER_STRENGTH, minimum=0.0, maximum=1.0)
APPLY_SPECTRAL_GATE = _bool_env("APPLY_SPECTRAL_GATE", _CFG_APPLY_SPECTRAL_GATE)
SPECTRAL_GATE_THRESHOLD = _float_env("SPECTRAL_GATE_THRESHOLD", _CFG_SPECTRAL_GATE_THRESHOLD, minimum=1.0)
APPLY_WIENER_POSTFILTER = _bool_env("APPLY_WIENER_POSTFILTER", _CFG_APPLY_WIENER_POSTFILTER)
WIENER_BETA = _float_env("WIENER_BETA", _CFG_WIENER_BETA, minimum=0.0)

# Griffin-Lim phase reconstruction — on by default to avoid noisy-phase bleed-through.
# Set USE_GRIFFIN_LIM=false to use noisy phase instead (faster but less clean).
USE_GRIFFIN_LIM = _bool_env("USE_GRIFFIN_LIM", True)
GRIFFIN_LIM_ITER = _int_env("GRIFFIN_LIM_ITER", _CFG_GRIFFIN_LIM_ITER, minimum=4)

# Residual suppression sharpens the model mask to attenuate low-confidence bins,
# which reduces faint background speech that survives denoising.
APPLY_RESIDUAL_SUPPRESS = _bool_env("APPLY_RESIDUAL_SUPPRESS", True)
RESIDUAL_SUPPRESS_POWER = _float_env("RESIDUAL_SUPPRESS_POWER", 2.2, minimum=1.0, maximum=4.0)
RESIDUAL_SUPPRESS_FLOOR = _float_env("RESIDUAL_SUPPRESS_FLOOR", 0.08, minimum=0.0, maximum=0.5)

# Speech-band suppression specifically targets voice-like residuals that remain
# after denoising (e.g., faint background conversation leakage).
APPLY_SPEECH_BAND_SUPPRESS = _bool_env("APPLY_SPEECH_BAND_SUPPRESS", True)
SPEECH_BAND_MIN_HZ = _float_env("SPEECH_BAND_MIN_HZ", 180.0, minimum=20.0)
SPEECH_BAND_MAX_HZ = _float_env("SPEECH_BAND_MAX_HZ", 4200.0, minimum=500.0)
SPEECH_SUPPRESS_THRESHOLD = _float_env("SPEECH_SUPPRESS_THRESHOLD", 0.62, minimum=0.05, maximum=0.95)
SPEECH_SUPPRESS_STEEPNESS = _float_env("SPEECH_SUPPRESS_STEEPNESS", 14.0, minimum=1.0, maximum=30.0)
SPEECH_SUPPRESS_FLOOR = _float_env("SPEECH_SUPPRESS_FLOOR", 0.12, minimum=0.0, maximum=0.8)
SPEECH_GAIN_SMOOTH = _int_env("SPEECH_GAIN_SMOOTH", 9, minimum=1)

# Frame-level gate in speech band: attenuates whole speech-band frames with
# low confidence, which is effective against faint background conversations.
APPLY_FRAME_SPEECH_GATE = _bool_env("APPLY_FRAME_SPEECH_GATE", True)
FRAME_SPEECH_GATE_THRESHOLD = _float_env("FRAME_SPEECH_GATE_THRESHOLD", 0.60, minimum=0.05, maximum=0.95)
FRAME_SPEECH_GATE_STEEPNESS = _float_env("FRAME_SPEECH_GATE_STEEPNESS", 18.0, minimum=1.0, maximum=40.0)
FRAME_SPEECH_GATE_FLOOR = _float_env("FRAME_SPEECH_GATE_FLOOR", 0.12, minimum=0.0, maximum=0.9)
FRAME_SPEECH_GATE_SMOOTH = _int_env("FRAME_SPEECH_GATE_SMOOTH", 13, minimum=1)
FRAME_SPEECH_GATE_DOWN_ALPHA = _float_env("FRAME_SPEECH_GATE_DOWN_ALPHA", 0.60, minimum=0.0, maximum=0.999)
FRAME_SPEECH_GATE_UP_ALPHA = _float_env("FRAME_SPEECH_GATE_UP_ALPHA", 0.92, minimum=0.0, maximum=0.999)

# Loudness matching restores speech audibility after suppression without hard
# peak-normalizing every output (which can also raise residual noise).
APPLY_LOUDNESS_MATCH = _bool_env("APPLY_LOUDNESS_MATCH", True)
LOUDNESS_TARGET_RATIO = _float_env("LOUDNESS_TARGET_RATIO", 0.90, minimum=0.4, maximum=1.2)
LOUDNESS_MAX_GAIN = _float_env("LOUDNESS_MAX_GAIN", 1.45, minimum=1.0, maximum=4.0)

# Denoise profile controls how much post-processing is applied after model
# prediction. "trained" stays closest to model behavior seen during training.
_DENOISE_MODE_RAW = os.getenv("DENOISE_MODE", "trained").strip().lower()
_VALID_DENOISE_MODES = {"trained", "balanced", "aggressive"}
if _DENOISE_MODE_RAW not in _VALID_DENOISE_MODES:
    logger.warning(
        "Invalid DENOISE_MODE=%r. Falling back to 'trained'. Valid options: %s",
        _DENOISE_MODE_RAW,
        sorted(_VALID_DENOISE_MODES),
    )
    DENOISE_MODE = "trained"
else:
    DENOISE_MODE = _DENOISE_MODE_RAW


def _load_audio_with_fallback(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio without hard dependency on TorchCodec.

    torchaudio>=2.8 may require torchcodec for decoding. If unavailable,
    we fall back to soundfile and then librosa.
    """
    torchaudio_error: Exception | None = None
    try:
        return torchaudio.load(str(path))
    except Exception as exc:
        torchaudio_error = exc
        logger.warning("torchaudio.load failed, trying fallback decoders: %s", exc)

    if sf is not None:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data).transpose(0, 1).contiguous()  # (C, T)
            return waveform, int(sr)
        except Exception as sf_exc:
            logger.warning("soundfile decode failed, trying librosa: %s", sf_exc)
    else:
        logger.warning("soundfile is not installed; trying librosa decoder")

    try:
        import librosa

        data, sr = librosa.load(str(path), sr=None, mono=False)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data)
        return waveform.float(), int(sr)
    except Exception as librosa_exc:
        raise RuntimeError(
            "Audio decoding failed. Install torchcodec (`pip install torchcodec`) "
            "or upload WAV/FLAC audio. "
            f"Last decoder error: {librosa_exc}"
        ) from (torchaudio_error or librosa_exc)


def _save_audio_with_fallback(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save audio with torchaudio first, then soundfile fallback."""
    try:
        torchaudio.save(str(path), waveform.cpu(), sample_rate)
        return
    except Exception as exc:
        logger.warning("torchaudio.save failed, trying soundfile: %s", exc)

    if sf is None:
        raise RuntimeError(
            "Saving audio failed because torchaudio.save is unavailable and soundfile "
            "is not installed. Install `soundfile` or `torchcodec`."
        )

    audio_np = waveform.detach().cpu().numpy().T  # (T, C)
    sf.write(str(path), audio_np, sample_rate)


async def _save_upload_stream(upload: UploadFile, destination: Path) -> None:
    """Stream uploaded file to disk to avoid loading full payload into memory."""
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            handle.write(chunk)


def _find_best_checkpoint() -> Path | None:
    """Pick checkpoint from MODEL_PATH override, then primary and fallback directories."""
    if MODEL_PATH:
        model_path = Path(MODEL_PATH).expanduser()
        if model_path.is_file():
            return model_path
        if model_path.is_dir():
            candidate = _pick_checkpoint_from_dir(model_path)
            if candidate is not None:
                return candidate
        logger.warning("MODEL_PATH did not resolve to a checkpoint: %s", model_path)

    for directory in (CHECKPOINT_DIR, FALLBACK_CHECKPOINT_DIR):
        candidate = _pick_checkpoint_from_dir(directory)
        if candidate is not None:
            return candidate
    return None


def _resolve_runtime_profile(duration_sec: float) -> tuple[str, bool, int]:
    """Adjust expensive inference steps for long clips to avoid timeouts/OOM."""
    mode = DENOISE_MODE
    use_griffin_lim = USE_GRIFFIN_LIM
    griffin_lim_iter = GRIFFIN_LIM_ITER

    if duration_sec >= LONG_AUDIO_SEC:
        if LONG_AUDIO_DISABLE_GRIFFIN_LIM:
            use_griffin_lim = False
        else:
            griffin_lim_iter = min(griffin_lim_iter, LONG_AUDIO_GRIFFIN_LIM_ITER)

        # Cap suppression profile for long clips to reduce processing overhead.
        if mode == "aggressive":
            mode = "balanced"

    return mode, use_griffin_lim, griffin_lim_iter


def _griffin_lim(
    magnitude: torch.Tensor,
    window: torch.Tensor,
    n_iter: int,
    init_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct waveform from magnitude-only spectrogram using Griffin-Lim algorithm.

    Phase is iteratively refined to be consistent with the clean magnitude.
    Using noisy input phase as initialization stabilizes reconstruction and
    reduces phase-related flutter artifacts.
    """
    if init_phase is not None and init_phase.shape == magnitude.shape:
        phase = init_phase
    else:
        phase = torch.rand_like(magnitude) * 2 * torch.pi - torch.pi
    for _ in range(n_iter):
        stft_est = magnitude * torch.exp(1j * phase)
        waveform = torch.istft(
            stft_est,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
        )
        stft_rebuilt = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            return_complex=True,
        )
        phase = torch.angle(stft_rebuilt)
    # Final reconstruction with converged phase
    stft_final = magnitude * torch.exp(1j * phase)
    return torch.istft(
        stft_final,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
    )


def _apply_residual_suppress(mag_clean: torch.Tensor, mag_noisy: torch.Tensor) -> torch.Tensor:
    """Attenuate low-confidence time-frequency bins that often carry background speech."""
    if not APPLY_RESIDUAL_SUPPRESS:
        return torch.clamp(mag_clean, min=0.0)

    eps = 1e-8
    mask = torch.clamp(mag_clean / (mag_noisy + eps), min=0.0, max=1.0)

    # Power>1 sharpens the mask by suppressing bins where clean/noisy ratio is low.
    attenuation = torch.pow(mask, RESIDUAL_SUPPRESS_POWER - 1.0)
    attenuation = RESIDUAL_SUPPRESS_FLOOR + (1.0 - RESIDUAL_SUPPRESS_FLOOR) * attenuation

    return torch.clamp(mag_clean * attenuation, min=0.0)


def _apply_speech_band_suppress(mag_clean: torch.Tensor, mag_noisy: torch.Tensor) -> torch.Tensor:
    """Suppress low-confidence bins in the speech band to reduce leaked voices."""
    if not APPLY_SPEECH_BAND_SUPPRESS:
        return torch.clamp(mag_clean, min=0.0)

    eps = 1e-8
    mask = torch.clamp(mag_clean / (mag_noisy + eps), min=0.0, max=1.0)

    # Smooth confidence over time to avoid rapid pumping artifacts.
    if mask.size(1) >= 5:
        mask = torch.nn.functional.avg_pool1d(mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

    keep = torch.sigmoid((mask - SPEECH_SUPPRESS_THRESHOLD) * SPEECH_SUPPRESS_STEEPNESS)
    keep = SPEECH_SUPPRESS_FLOOR + (1.0 - SPEECH_SUPPRESS_FLOOR) * keep

    kernel = SPEECH_GAIN_SMOOTH
    if kernel % 2 == 0:
        kernel += 1
    if kernel > 1 and keep.size(1) >= kernel:
        keep = torch.nn.functional.avg_pool1d(
            keep.unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).squeeze(0)

    nyquist = SAMPLE_RATE / 2.0
    min_hz = max(0.0, min(SPEECH_BAND_MIN_HZ, nyquist))
    max_hz = max(min_hz, min(SPEECH_BAND_MAX_HZ, nyquist))
    freqs = torch.linspace(0.0, nyquist, mag_clean.size(0), device=mag_clean.device, dtype=mag_clean.dtype).unsqueeze(1)
    band = ((freqs >= min_hz) & (freqs <= max_hz)).to(mag_clean.dtype)

    gain = (1.0 - band) + (band * keep)
    return torch.clamp(mag_clean * gain, min=0.0)


def _apply_frame_speech_gate(mag_clean: torch.Tensor, mag_noisy: torch.Tensor) -> torch.Tensor:
    """Suppress whole speech-band frames when confidence stays low over time."""
    if not APPLY_FRAME_SPEECH_GATE:
        return torch.clamp(mag_clean, min=0.0)

    eps = 1e-8
    mask = torch.clamp(mag_clean / (mag_noisy + eps), min=0.0, max=1.0)

    nyquist = SAMPLE_RATE / 2.0
    min_hz = max(0.0, min(SPEECH_BAND_MIN_HZ, nyquist))
    max_hz = max(min_hz, min(SPEECH_BAND_MAX_HZ, nyquist))
    freqs = torch.linspace(0.0, nyquist, mag_clean.size(0), device=mag_clean.device, dtype=mag_clean.dtype).unsqueeze(1)
    band = ((freqs >= min_hz) & (freqs <= max_hz)).to(mag_clean.dtype)

    band_bins = band.sum().clamp_min(1.0)
    frame_conf = (mask * band).sum(dim=0, keepdim=True) / band_bins

    kernel = FRAME_SPEECH_GATE_SMOOTH
    if kernel % 2 == 0:
        kernel += 1
    if kernel > 1 and frame_conf.size(1) >= kernel:
        frame_conf = torch.nn.functional.avg_pool1d(
            frame_conf.unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).squeeze(0)

    frame_keep = torch.sigmoid((frame_conf - FRAME_SPEECH_GATE_THRESHOLD) * FRAME_SPEECH_GATE_STEEPNESS)
    frame_keep = FRAME_SPEECH_GATE_FLOOR + (1.0 - FRAME_SPEECH_GATE_FLOOR) * frame_keep

    # Attack/release smoothing: suppress quickly, recover gradually, reducing pumping.
    if frame_keep.size(1) > 1:
        down_alpha = torch.tensor(FRAME_SPEECH_GATE_DOWN_ALPHA, device=frame_keep.device, dtype=frame_keep.dtype)
        up_alpha = torch.tensor(FRAME_SPEECH_GATE_UP_ALPHA, device=frame_keep.device, dtype=frame_keep.dtype)
        smoothed = frame_keep.clone()
        prev = smoothed[:, :1]
        for idx in range(1, smoothed.size(1)):
            target = smoothed[:, idx:idx + 1]
            alpha = torch.where(target < prev, down_alpha, up_alpha)
            prev = (alpha * prev) + ((1.0 - alpha) * target)
            smoothed[:, idx:idx + 1] = prev
        frame_keep = smoothed

    gain = (1.0 - band) + (band * frame_keep)
    return torch.clamp(mag_clean * gain, min=0.0)


def _apply_loudness_match(waveform_clean: torch.Tensor, waveform_input: torch.Tensor) -> torch.Tensor:
    """Apply capped RMS gain to keep denoised speech audible."""
    if not APPLY_LOUDNESS_MATCH:
        return waveform_clean

    eps = 1e-8
    in_rms = torch.sqrt(torch.mean(waveform_input.pow(2)) + eps)
    out_rms = torch.sqrt(torch.mean(waveform_clean.pow(2)) + eps)

    if torch.isnan(in_rms) or torch.isnan(out_rms) or in_rms <= 0 or out_rms <= 0:
        return waveform_clean

    target_rms = in_rms * LOUDNESS_TARGET_RATIO
    gain = torch.clamp(target_rms / (out_rms + eps), min=1.0, max=LOUDNESS_MAX_GAIN)
    return waveform_clean * gain


def _estimate_noise_floor(magnitude: torch.Tensor) -> torch.Tensor:
    """Estimate stationary background floor per frequency bin."""
    try:
        floor = torch.quantile(magnitude, q=0.10, dim=1, keepdim=True)
    except Exception:
        floor = torch.median(magnitude, dim=1, keepdim=True).values
    return floor.clamp_min(1e-8)


def _apply_postfilter(magnitude: torch.Tensor) -> torch.Tensor:
    """Apply a light post-filter to suppress residual hiss without over-suppressing speech.

    The spectral gate mask is smoothed over neighbouring time frames to prevent
    rapid on/off switching, which causes musical noise / repetition artifacts.
    """
    if not APPLY_POSTFILTER:
        return torch.clamp(magnitude, min=0.0)

    filtered = torch.clamp(magnitude, min=0.0)
    noise_floor = _estimate_noise_floor(filtered)

    if APPLY_SPECTRAL_GATE:
        threshold = noise_floor * SPECTRAL_GATE_THRESHOLD
        # Compute soft gate in [0, 1] — steeper slope = sharper cutoff
        gate = torch.sigmoid(((filtered / (threshold + 1e-8)) - 1.0) * 4.0)

        # Temporally smooth the gate mask with a small median over time to avoid
        # isolated surviving bins (the main cause of musical noise / "repetition")
        if gate.size(1) >= 5:
            # Unfold time dimension into a sliding window and take median per bin
            kernel = 5
            padded = torch.nn.functional.pad(gate, (kernel // 2, kernel // 2), mode="reflect")
            # shape: (F, T, kernel)
            windows = padded.unfold(1, kernel, 1)
            gate = windows.median(dim=-1).values

        filtered = filtered * gate

    if APPLY_WIENER_POSTFILTER:
        signal_power = filtered.pow(2)
        noise_power = noise_floor.pow(2) * (1.0 + WIENER_BETA)
        wiener_gain = signal_power / (signal_power + noise_power + 1e-8)
        # Smooth Wiener gain over time to reduce rapid gain fluctuations
        if wiener_gain.size(1) >= 3:
            wiener_gain = torch.nn.functional.avg_pool1d(
                wiener_gain,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        filtered = filtered * wiener_gain

    nyquist = SAMPLE_RATE / 2.0
    cutoff_hz = min(POSTFILTER_CUTOFF_HZ, nyquist)
    if cutoff_hz < nyquist and POSTFILTER_STRENGTH > 0.0:
        freqs = torch.linspace(0.0, nyquist, filtered.size(0), device=filtered.device, dtype=filtered.dtype).unsqueeze(1)
        transition_hz = 500.0
        lowpass_curve = torch.clamp((cutoff_hz + transition_hz - freqs) / transition_hz, 0.0, 1.0)
        keep = (1.0 - POSTFILTER_STRENGTH) + (POSTFILTER_STRENGTH * lowpass_curve)
        filtered = filtered * keep

    return torch.clamp(filtered, min=0.0)


def _compute_ssim_1d(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Global SSIM approximation for 1D waveforms."""
    if reference.numel() < 2 or estimate.numel() < 2:
        return 1.0

    x = reference.float()
    y = estimate.float()

    c1 = 1e-4
    c2 = 9e-4

    mu_x = x.mean()
    mu_y = y.mean()
    var_x = torch.mean((x - mu_x).pow(2))
    var_y = torch.mean((y - mu_y).pow(2))
    cov_xy = torch.mean((x - mu_x) * (y - mu_y))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    if float(denominator) <= 0.0:
        return 1.0

    value = float((numerator / denominator).item())
    return max(-1.0, min(1.0, value))


def _compute_lsd(mag_a: torch.Tensor, mag_b: torch.Tensor) -> float:
    """Log-spectral distance over full STFT magnitude tensors."""
    if mag_a.numel() == 0 or mag_b.numel() == 0:
        return 0.0

    freq_bins = min(mag_a.size(0), mag_b.size(0))
    time_bins = min(mag_a.size(1), mag_b.size(1))
    if freq_bins <= 0 or time_bins <= 0:
        return 0.0
    mag_a = mag_a[:freq_bins, :time_bins]
    mag_b = mag_b[:freq_bins, :time_bins]

    eps = 1e-8
    log_a = torch.log10(torch.clamp(mag_a, min=eps))
    log_b = torch.log10(torch.clamp(mag_b, min=eps))
    diff = log_a - log_b
    return float(torch.sqrt(torch.mean(diff.pow(2))).item())


def _compute_proxy_metrics(
    waveform_noisy: torch.Tensor,
    waveform_clean: torch.Tensor,
    mag_noisy: torch.Tensor,
    mag_clean: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Compute bidirectional proxy metrics from noisy and denoised audio.

    `before` uses denoised audio as reference and evaluates noisy audio.
    `after` uses noisy audio as reference and evaluates denoised audio.
    """

    def _directional_metrics(
        reference_waveform: torch.Tensor,
        estimate_waveform: torch.Tensor,
        reference_mag: torch.Tensor,
        estimate_mag: torch.Tensor,
    ) -> dict[str, float]:
        eps = 1e-8

        reference = reference_waveform.detach().float().flatten()
        estimate = estimate_waveform.detach().float().flatten()
        n = min(reference.numel(), estimate.numel())
        if n <= 1:
            return {"snr": 0.0, "psnr": 0.0, "ssim": 0.0, "lsd": 0.0}

        reference = reference[:n]
        estimate = estimate[:n]
        error = reference - estimate

        mse = float(torch.mean(error.pow(2)).item())
        ref_power = float(torch.mean(reference.pow(2)).item())
        peak = max(float(torch.max(reference.abs()).item()), eps)

        snr = 10.0 * math.log10((ref_power + eps) / (mse + eps))
        psnr = 10.0 * math.log10((peak * peak + eps) / (mse + eps))
        ssim = _compute_ssim_1d(reference, estimate)
        lsd = _compute_lsd(
            reference_mag.detach().float(),
            estimate_mag.detach().float(),
        )

        return {
            "snr": max(-120.0, min(120.0, snr)),
            "psnr": max(-120.0, min(120.0, psnr)),
            "ssim": max(-1.0, min(1.0, ssim)),
            "lsd": max(0.0, lsd),
        }

    return {
        "before": _directional_metrics(
            waveform_clean,
            waveform_noisy,
            mag_clean,
            mag_noisy,
        ),
        "after": _directional_metrics(
            waveform_noisy,
            waveform_clean,
            mag_noisy,
            mag_clean,
        ),
    }


def _load_model() -> UNet | None:
    ckpt_path = _find_best_checkpoint()
    if ckpt_path is None:
        logger.warning("No checkpoint found in %s", CHECKPOINT_DIR)
        return None

    logger.info("Loading checkpoint: %s  (device=%s)", ckpt_path, _device)

    final_activation = "none" if LOSS_FUNCTION == "crm" else "softplus"
    model = UNet(
        in_ch=UNET_IN_CH,
        out_ch=UNET_OUT_CH,
        base_ch=UNET_BASE_CH,
        final_activation=final_activation,
        mask_mode=MASK_MODE,
    )

    state = torch.load(ckpt_path, map_location=_device, weights_only=False)
    # handle checkpoints that wrap state_dict inside a dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(_device).eval()
    logger.info(
        "Inference settings: use_log_mag=%s, mask_mode=%s, loss_function=%s",
        USE_LOG_MAG,
        MASK_MODE,
        LOSS_FUNCTION,
    )
    logger.info(
        "Post-filter settings: enabled=%s, spectral_gate=%s(threshold=%.2f), wiener=%s(beta=%.3f), lowpass_cutoff=%.0fHz(strength=%.2f)",
        APPLY_POSTFILTER,
        APPLY_SPECTRAL_GATE,
        SPECTRAL_GATE_THRESHOLD,
        APPLY_WIENER_POSTFILTER,
        WIENER_BETA,
        POSTFILTER_CUTOFF_HZ,
        POSTFILTER_STRENGTH,
    )
    logger.info(
        "Phase/residual settings: griffin_lim=%s(iter=%d), residual_suppress=%s(power=%.2f,floor=%.2f)",
        USE_GRIFFIN_LIM,
        GRIFFIN_LIM_ITER,
        APPLY_RESIDUAL_SUPPRESS,
        RESIDUAL_SUPPRESS_POWER,
        RESIDUAL_SUPPRESS_FLOOR,
    )
    logger.info(
        "Speech-band suppress: enabled=%s, band=%.0f-%.0fHz, threshold=%.2f, steepness=%.1f, floor=%.2f, smooth=%d",
        APPLY_SPEECH_BAND_SUPPRESS,
        SPEECH_BAND_MIN_HZ,
        SPEECH_BAND_MAX_HZ,
        SPEECH_SUPPRESS_THRESHOLD,
        SPEECH_SUPPRESS_STEEPNESS,
        SPEECH_SUPPRESS_FLOOR,
        SPEECH_GAIN_SMOOTH,
    )
    logger.info(
        "Frame speech gate: enabled=%s, threshold=%.2f, steepness=%.1f, floor=%.2f, smooth=%d, down_alpha=%.2f, up_alpha=%.2f",
        APPLY_FRAME_SPEECH_GATE,
        FRAME_SPEECH_GATE_THRESHOLD,
        FRAME_SPEECH_GATE_STEEPNESS,
        FRAME_SPEECH_GATE_FLOOR,
        FRAME_SPEECH_GATE_SMOOTH,
        FRAME_SPEECH_GATE_DOWN_ALPHA,
        FRAME_SPEECH_GATE_UP_ALPHA,
    )
    logger.info(
        "Loudness match: enabled=%s, target_ratio=%.2f, max_gain=%.2f",
        APPLY_LOUDNESS_MATCH,
        LOUDNESS_TARGET_RATIO,
        LOUDNESS_MAX_GAIN,
    )
    logger.info("Denoise mode: %s (trained|balanced|aggressive)", DENOISE_MODE)
    logger.info(
        "Long-audio safety: threshold=%ss, disable_griffin_lim=%s, fallback_iter=%d, upload_chunk_bytes=%d",
        LONG_AUDIO_SEC,
        LONG_AUDIO_DISABLE_GRIFFIN_LIM,
        LONG_AUDIO_GRIFFIN_LIM_ITER,
        UPLOAD_CHUNK_BYTES,
    )
    logger.info(
        "Thread settings: torch_threads=%d, torch_interop_threads=%d",
        torch.get_num_threads(),
        torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else -1,
    )
    logger.info("Model loaded successfully (%d params)", sum(p.numel() for p in model.parameters()))
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model
    _model = _load_model()
    yield
    _model = None


# ── FastAPI App ──
app = FastAPI(
    title="DenoiseAI API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Processing-Time", *METRIC_HEADER_NAMES],
)


# ──────────────────── ROUTES ────────────────────


@app.get("/")
def read_root():
    return {"message": "Welcome to DenoiseAI — use /api/health or POST /api/denoise"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if _model is not None else "error",
        "model_loaded": _model is not None,
        "device": str(_device),
        "use_griffin_lim": USE_GRIFFIN_LIM,
        "griffin_lim_iter": GRIFFIN_LIM_ITER,
        "apply_residual_suppress": APPLY_RESIDUAL_SUPPRESS,
        "apply_speech_band_suppress": APPLY_SPEECH_BAND_SUPPRESS,
        "apply_frame_speech_gate": APPLY_FRAME_SPEECH_GATE,
        "apply_loudness_match": APPLY_LOUDNESS_MATCH,
        "loudness_target_ratio": LOUDNESS_TARGET_RATIO,
        "denoise_mode": DENOISE_MODE,
        "long_audio_sec": LONG_AUDIO_SEC,
        "long_audio_disable_griffin_lim": LONG_AUDIO_DISABLE_GRIFFIN_LIM,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else None,
        "version": "1.0.0",
    }


def _denoise_file_sync(
    tmp_input: Path,
    original_filename: str,
    uid: str,
) -> tuple[Path, float, str, dict[str, dict[str, float]]]:
    """CPU-heavy denoising path executed in a worker thread."""
    t0 = time.perf_counter()

    # Load audio
    waveform, sr = _load_audio_with_fallback(tmp_input)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (T,)

    # Duration check
    dur = waveform.shape[0] / SAMPLE_RATE
    if dur > MAX_DURATION_SEC:
        raise HTTPException(400, detail=f"Audio too long ({dur:.1f}s, max {MAX_DURATION_SEC}s).")

    runtime_mode, runtime_use_griffin_lim, runtime_griffin_lim_iter = _resolve_runtime_profile(dur)
    logger.info(
        "Runtime profile: duration=%.1fs, mode=%s, griffin_lim=%s, iter=%d",
        dur,
        runtime_mode,
        runtime_use_griffin_lim,
        runtime_griffin_lim_iter,
    )

    # STFT
    window = (_CFG_WINDOW if _CFG_WINDOW is not None else torch.hann_window(WIN_LENGTH)).to(_device)
    stft = torch.stft(
        waveform.to(_device),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(stft)       # (F, T)
    phase = torch.angle(stft)   # (F, T)

    # Match training input domain: model expects log-magnitude when USE_LOG_MAG=True.
    model_input = torch.log1p(mag) if USE_LOG_MAG else mag

    # Model forward pass
    mag_input = model_input.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
    with torch.no_grad():
        mag_clean = _model(mag_input)  # (1, 1, F, T)

    if mag_clean.dim() != 4 or mag_clean.size(1) != 1:
        raise RuntimeError(
            f"Unsupported model output shape {tuple(mag_clean.shape)}. "
            "Backend currently expects 1-channel magnitude output."
        )

    mag_clean = mag_clean.squeeze(0).squeeze(0)  # (F, T)

    # Convert back to linear magnitude before ISTFT when model output is log-magnitude.
    if USE_LOG_MAG:
        mag_clean = torch.expm1(mag_clean)
    mag_clean = torch.clamp(mag_clean, min=0.0)

    # Optional post-filter to reduce residual hiss after model inference.
    mag_clean = _apply_postfilter(mag_clean)

    # Profile-based suppression stack:
    # - trained: closest to model output (least speech damage)
    # - balanced: moderate residual suppression
    # - aggressive: strongest background suppression
    if runtime_mode in {"balanced", "aggressive"}:
        mag_clean = _apply_residual_suppress(mag_clean, mag)
        mag_clean = _apply_speech_band_suppress(mag_clean, mag)
    if runtime_mode == "aggressive":
        mag_clean = _apply_frame_speech_gate(mag_clean, mag)

    # Phase reconstruction — Griffin-Lim iterates to find a phase consistent
    # with the clean magnitude, eliminating noisy-phase bleed-through.
    # Falls back to noisy-phase ISTFT when use_griffin_lim=false (faster).
    if runtime_use_griffin_lim:
        waveform_clean = _griffin_lim(mag_clean, window, runtime_griffin_lim_iter, init_phase=phase)
    else:
        stft_clean = mag_clean * torch.exp(1j * phase)
        waveform_clean = torch.istft(
            stft_clean,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
        )

    waveform_clean = _apply_loudness_match(waveform_clean, waveform.to(waveform_clean.device))

    # Normalize only when needed to avoid re-amplifying residual background.
    peak = waveform_clean.abs().max()
    if peak > 0.95:
        waveform_clean = waveform_clean / peak * 0.95

    noisy_for_metrics = waveform.to(waveform_clean.device)
    noisy_stft_metric = torch.stft(
        noisy_for_metrics,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    clean_stft_metric = torch.stft(
        waveform_clean,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    metrics = _compute_proxy_metrics(
        noisy_for_metrics,
        waveform_clean,
        torch.abs(noisy_stft_metric),
        torch.abs(clean_stft_metric),
    )

    # Save output
    out_path = TEMP_DIR / f"denoised_{uid}.wav"
    _save_audio_with_fallback(out_path, waveform_clean.unsqueeze(0), SAMPLE_RATE)

    elapsed = time.perf_counter() - t0
    logger.info("Denoised %s in %.2fs", original_filename, elapsed)
    output_name = f"denoised_{Path(original_filename or 'audio').stem}.wav"
    return out_path, elapsed, output_name, metrics


@app.post("/api/denoise")
async def denoise(file: UploadFile = File(...)):
    """Accept an audio file, denoise it, and return the cleaned WAV."""

    if _model is None:
        raise HTTPException(503, detail="Model not loaded — check server logs.")

    # Validate extension
    ext = Path(file.filename or "upload.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            detail=f"Unsupported format '{ext}'. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Save upload to temp file
    uid = uuid.uuid4().hex[:12]
    tmp_input = TEMP_DIR / f"input_{uid}{ext}"
    try:
        await _save_upload_stream(file, tmp_input)
    except Exception as exc:
        raise HTTPException(500, detail=f"Failed to save upload: {exc}")

    try:
        out_path, elapsed, output_name, metrics = await run_in_threadpool(
            _denoise_file_sync,
            tmp_input,
            file.filename or "audio",
            uid,
        )

        headers = {"X-Processing-Time": f"{elapsed:.3f}"}
        for phase_key, prefix in (("before", "X-Metric-Before"), ("after", "X-Metric-After")):
            phase_metrics = metrics.get(phase_key, {})
            for metric_name in ("snr", "psnr", "ssim", "lsd"):
                value = phase_metrics.get(metric_name)
                if value is None:
                    continue
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    continue
                header_name = f"{prefix}-{metric_name.upper()}"
                headers[header_name] = f"{numeric_value:.6f}"

        return FileResponse(
            str(out_path),
            media_type="audio/wav",
            filename=output_name,
            headers=headers,
            background=BackgroundTask(out_path.unlink, missing_ok=True),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Denoising failed")
        raise HTTPException(500, detail=f"Processing error: {exc}")
    finally:
        # clean up input
        tmp_input.unlink(missing_ok=True)