"""FastAPI backend for audio denoising.

Serves the U-Net model and provides:
  POST /api/denoise  — accepts audio file, returns denoised WAV
  GET  /api/health   — liveness / readiness check
"""

import os
import time
import uuid
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

logger = logging.getLogger("denoiser")
logging.basicConfig(level=logging.INFO)


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

# ── Accepted audio extensions ──
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".webm"}
MAX_DURATION_SEC = _int_env("MAX_DURATION_SEC", 325, minimum=10)

# Optional post-filter controls (env overrides config defaults).
APPLY_POSTFILTER = _bool_env("APPLY_POSTFILTER", _CFG_APPLY_POSTFILTER)
POSTFILTER_CUTOFF_HZ = _float_env("POSTFILTER_CUTOFF_HZ", _CFG_POSTFILTER_CUTOFF_HZ, minimum=1000.0)
POSTFILTER_STRENGTH = _float_env("POSTFILTER_STRENGTH", _CFG_POSTFILTER_STRENGTH, minimum=0.0, maximum=1.0)
APPLY_SPECTRAL_GATE = _bool_env("APPLY_SPECTRAL_GATE", _CFG_APPLY_SPECTRAL_GATE)
SPECTRAL_GATE_THRESHOLD = _float_env("SPECTRAL_GATE_THRESHOLD", _CFG_SPECTRAL_GATE_THRESHOLD, minimum=1.0)
APPLY_WIENER_POSTFILTER = _bool_env("APPLY_WIENER_POSTFILTER", _CFG_APPLY_WIENER_POSTFILTER)
WIENER_BETA = _float_env("WIENER_BETA", _CFG_WIENER_BETA, minimum=0.0)


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
    expose_headers=["X-Processing-Time"],
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
        "version": "1.0.0",
    }


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
        contents = await file.read()
        tmp_input.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(500, detail=f"Failed to save upload: {exc}")

    t0 = time.perf_counter()

    try:
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

        # iSTFT
        stft_clean = mag_clean * torch.exp(1j * phase)
        waveform_clean = torch.istft(
            stft_clean,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
        )

        # Normalize to prevent clipping
        peak = waveform_clean.abs().max()
        if peak > 0:
            waveform_clean = waveform_clean / peak * 0.95

        # Save output
        out_path = TEMP_DIR / f"denoised_{uid}.wav"
        _save_audio_with_fallback(out_path, waveform_clean.unsqueeze(0), SAMPLE_RATE)

        elapsed = time.perf_counter() - t0
        logger.info("Denoised %s in %.2fs", file.filename, elapsed)

        return FileResponse(
            str(out_path),
            media_type="audio/wav",
            filename=f"denoised_{Path(file.filename or 'audio').stem}.wav",
            headers={"X-Processing-Time": f"{elapsed:.3f}"},
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