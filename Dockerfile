FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	OMP_NUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 \
	TORCH_THREADS=1 \
	TORCH_INTEROP_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg \
	libsndfile1 \
	wget \
	&& rm -rf /var/lib/apt/lists/*

# Install backend runtime dependencies.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# Copy backend code plus project-level model/config modules.
COPY backend/app /app/backend/app
COPY config.py /app/config.py
COPY model.py /app/model.py
COPY utilis.py /app/utilis.py

# Download model from Hugging Face Model Hub at build time.
RUN mkdir -p /app/backend/models && \
	wget -q -O /app/backend/models/unet_best.pt \
	https://huggingface.co/sujan12321/audiodenoising-model/resolve/main/unet_best.pt

EXPOSE 7860

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
