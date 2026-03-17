---
title: Audio Denoising
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Audio Denoising

A full-stack deep-learning app for speech denoising with a React frontend and a FastAPI backend running a U-Net model in the STFT domain.

## Live Deployment

- Frontend (Vercel): https://audiodenoising.vercel.app
- Backend (Hugging Face Space): https://sujan12321-audiodenoising.hf.space
- Health check: https://sujan12321-audiodenoising.hf.space/api/health

## Features

- Upload noisy audio and run denoising in one click
- Waveform and spectrogram comparison UI
- Production-ready FastAPI inference endpoint
- Docker-based backend deployment on Hugging Face Spaces

## Tech Stack

- Frontend: React, TypeScript, Vite
- Backend: FastAPI, PyTorch, torchaudio/librosa/soundfile
- Model: U-Net for magnitude-domain denoising
- Hosting: Vercel (frontend) + Hugging Face Space Docker (backend)

## Local Development

### Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

### Frontend (Vercel)

- `VITE_API_BASE` (example: `https://sujan12321-audiodenoising.hf.space/api`)
- `VITE_API_TIMEOUT_MS` (example: `300000`)

### Backend (Hugging Face Space Variables)

- `MODEL_PATH=/app/backend/models/unet_best.pt`
- `MAX_DURATION_SEC=300`
- `ALLOWED_ORIGINS=https://audiodenoising.vercel.app,http://localhost:3000`

Optional residual-noise filter tuning:
- `APPLY_POSTFILTER=true`
- `APPLY_SPECTRAL_GATE=true`
- `SPECTRAL_GATE_THRESHOLD=1.5`
- `APPLY_WIENER_POSTFILTER=true`
- `WIENER_BETA=0.02`
- `POSTFILTER_CUTOFF_HZ=6500`
- `POSTFILTER_STRENGTH=0.35`

## Deployment Notes

- Use `https://audiodenoising.vercel.app` as the stable production frontend URL. Deployment-specific Vercel URLs with random suffixes are snapshots and do not update in place.
- The repo root now includes `vercel.json` so Vercel can build the frontend from `frontend/` even when the project root is set to the repository root.
- Large model binaries are not committed to GitHub because files over 100MB are blocked.
- The Docker build downloads the model from Hugging Face Model Hub at build time.
- If frontend URL changes, update `ALLOWED_ORIGINS` in the Hugging Face Space settings.

## License

Apache-2.0