FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg \
	libsndfile1 \
	&& rm -rf /var/lib/apt/lists/*

# Install backend runtime dependencies.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# Copy backend code plus project-level model/config modules used by backend/app/main.py.
COPY backend/app /app/backend/app
COPY backend/models /app/backend/models
COPY config.py /app/config.py
COPY model.py /app/model.py
COPY utilis.py /app/utilis.py

# Model checkpoints should be mounted or downloaded at runtime.
RUN mkdir -p /app/checkpoints

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
