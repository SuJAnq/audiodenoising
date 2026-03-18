import type {
  DenoiseMetrics,
  DenoiseResponse,
  HealthStatus,
} from "../types/audio";

/**
 * Base URL for API calls.
 * In dev the Vite proxy forwards /api → http://localhost:8000
 * In production, set VITE_API_BASE to the actual backend URL.
 */
const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

const REQUEST_TIMEOUT_MS = (() => {
  const raw = import.meta.env.VITE_API_TIMEOUT_MS;
  const parsed = Number(raw);
  if (Number.isFinite(parsed) && parsed > 0) return parsed;
  return 300_000; // 5 minutes default for longer clips on free-tier CPUs
})();

if (import.meta.env.PROD && !import.meta.env.VITE_API_BASE) {
  console.warn(
    "VITE_API_BASE is not set. API requests will use '/api', which requires reverse proxy routing in production.",
  );
}

const METRIC_NAMES = ["SNR", "PSNR", "SSIM", "LSD"] as const;
const MAX_METRIC_SAMPLES = 48_000;
const EPSILON = 1e-8;

const hasAnyMetric = (metrics?: DenoiseMetrics): boolean =>
  metrics?.snr != null ||
  metrics?.psnr != null ||
  metrics?.ssim != null ||
  metrics?.lsd != null;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const isMetricShape = (value: unknown): value is DenoiseMetrics =>
  !!value &&
  typeof value === "object" &&
  ("snr" in value ||
    "psnr" in value ||
    "ssim" in value ||
    "lsd" in value);

function responseHasAnyMetrics(data: DenoiseResponse): boolean {
  const metricsRaw = data.metrics;
  const metricsRecord =
    metricsRaw && typeof metricsRaw === "object"
      ? (metricsRaw as Record<string, unknown>)
      : undefined;

  const candidates: Array<DenoiseMetrics | undefined> = [
    data.before_metrics,
    data.after_metrics,
    data.noisy_metrics,
    data.denoised_metrics,
    data.metrics_before,
    data.metrics_after,
    isMetricShape(metricsRaw) ? metricsRaw : undefined,
    isMetricShape(metricsRecord?.before) ? metricsRecord?.before : undefined,
    isMetricShape(metricsRecord?.after) ? metricsRecord?.after : undefined,
    isMetricShape(metricsRecord?.noisy) ? metricsRecord?.noisy : undefined,
    isMetricShape(metricsRecord?.denoised)
      ? metricsRecord?.denoised
      : undefined,
  ];

  return candidates.some((metric) => hasAnyMetric(metric));
}

function downsampleForMetrics(signal: Float32Array): Float32Array {
  if (signal.length <= MAX_METRIC_SAMPLES) return signal;

  const stride = Math.ceil(signal.length / MAX_METRIC_SAMPLES);
  const outLength = Math.ceil(signal.length / stride);
  const out = new Float32Array(outLength);

  let j = 0;
  for (let i = 0; i < signal.length && j < outLength; i += stride) {
    out[j] = signal[i];
    j += 1;
  }

  return out;
}

async function decodeAudioToMono(blob: Blob): Promise<Float32Array> {
  const AudioContextCtor =
    window.AudioContext ||
    (
      window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext }
    ).webkitAudioContext;

  if (!AudioContextCtor) {
    throw new Error("Web Audio API is unavailable in this browser.");
  }

  const audioContext = new AudioContextCtor();

  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    // Use first channel for lightweight proxy metrics.
    return new Float32Array(audioBuffer.getChannelData(0));
  } finally {
    await audioContext.close();
  }
}

function computeProxyMetricsFromWaveforms(
  noisyInput: Float32Array,
  cleanInput: Float32Array,
): Pick<DenoiseResponse, "before_metrics" | "after_metrics"> | undefined {
  const noisy = downsampleForMetrics(noisyInput);
  const clean = downsampleForMetrics(cleanInput);

  const n = Math.min(noisy.length, clean.length);
  if (n < 2) return undefined;

  const noisyTrimmed = noisy.subarray(0, n);
  const cleanTrimmed = clean.subarray(0, n);

  const directionalMetrics = (
    reference: Float32Array,
    estimate: Float32Array,
  ): DenoiseMetrics => {
    let sumErrorSq = 0;
    let sumReferenceSq = 0;
    let peakReference = 0;
    let sumReference = 0;
    let sumEstimate = 0;
    let sumLogDiffSq = 0;

    for (let i = 0; i < n; i += 1) {
      const referenceValue = reference[i];
      const estimateValue = estimate[i];
      const error = referenceValue - estimateValue;

      sumErrorSq += error * error;
      sumReferenceSq += referenceValue * referenceValue;
      peakReference = Math.max(peakReference, Math.abs(referenceValue));
      sumReference += referenceValue;
      sumEstimate += estimateValue;

      const logReference = Math.log10(
        Math.max(Math.abs(referenceValue), EPSILON),
      );
      const logEstimate = Math.log10(Math.max(Math.abs(estimateValue), EPSILON));
      const logDiff = logReference - logEstimate;
      sumLogDiffSq += logDiff * logDiff;
    }

    const mse = sumErrorSq / n;
    const referencePower = sumReferenceSq / n;
    const peak = Math.max(peakReference, EPSILON);

    const snr = 10 * Math.log10((referencePower + EPSILON) / (mse + EPSILON));
    const psnr = 10 * Math.log10((peak * peak + EPSILON) / (mse + EPSILON));
    const lsd = Math.sqrt(sumLogDiffSq / n);

    const muReference = sumReference / n;
    const muEstimate = sumEstimate / n;

    let varReference = 0;
    let varEstimate = 0;
    let covariance = 0;

    for (let i = 0; i < n; i += 1) {
      const referenceCentered = reference[i] - muReference;
      const estimateCentered = estimate[i] - muEstimate;
      varReference += referenceCentered * referenceCentered;
      varEstimate += estimateCentered * estimateCentered;
      covariance += referenceCentered * estimateCentered;
    }

    varReference /= n;
    varEstimate /= n;
    covariance /= n;

    const c1 = 1e-4;
    const c2 = 9e-4;
    const ssimNumerator =
      (2 * muReference * muEstimate + c1) * (2 * covariance + c2);
    const ssimDenominator =
      (muReference * muReference + muEstimate * muEstimate + c1) *
      (varReference + varEstimate + c2);
    const ssim =
      ssimDenominator > 0
        ? clamp(ssimNumerator / ssimDenominator, -1, 1)
        : 1;

    return {
      snr: clamp(snr, -120, 120),
      psnr: clamp(psnr, -120, 120),
      ssim,
      lsd: Math.max(0, lsd),
    };
  };

  return {
    before_metrics: directionalMetrics(cleanTrimmed, noisyTrimmed),
    after_metrics: directionalMetrics(noisyTrimmed, cleanTrimmed),
  };
}

async function computeProxyMetricsFromAudio(
  originalFile: File,
  denoisedBlob: Blob,
): Promise<Pick<DenoiseResponse, "before_metrics" | "after_metrics"> | undefined> {
  try {
    const [noisy, clean] = await Promise.all([
      decodeAudioToMono(originalFile),
      decodeAudioToMono(denoisedBlob),
    ]);
    return computeProxyMetricsFromWaveforms(noisy, clean);
  } catch (error) {
    console.warn("Failed to compute client-side proxy metrics:", error);
    return undefined;
  }
}

function parseMetricGroupFromHeaders(
  xhr: XMLHttpRequest,
  prefix: "X-Metric-Before" | "X-Metric-After",
): DenoiseMetrics | undefined {
  const metrics: DenoiseMetrics = {};

  for (const metricName of METRIC_NAMES) {
    const raw = xhr.getResponseHeader(`${prefix}-${metricName}`);
    if (!raw) continue;

    const value = Number(raw);
    if (!Number.isFinite(value)) continue;

    const key = metricName.toLowerCase() as keyof DenoiseMetrics;
    metrics[key] = value;
  }

  return Object.keys(metrics).length > 0 ? metrics : undefined;
}

function parseProxyMetricsFromHeaders(
  xhr: XMLHttpRequest,
): Pick<DenoiseResponse, "before_metrics" | "after_metrics"> {
  const beforeMetrics = parseMetricGroupFromHeaders(xhr, "X-Metric-Before");
  const afterMetrics = parseMetricGroupFromHeaders(xhr, "X-Metric-After");

  return {
    ...(beforeMetrics ? { before_metrics: beforeMetrics } : {}),
    ...(afterMetrics ? { after_metrics: afterMetrics } : {}),
  };
}

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Check backend health / model readiness.
 */
export async function checkHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return { status: "error", model_loaded: false };
    return await res.json();
  } catch {
    return { status: "error", model_loaded: false };
  }
}

/**
 * Upload a noisy audio file and receive the denoised result.
 * Returns the parsed response and a blob URL for playback.
 */
export async function denoiseAudio(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<{ data: DenoiseResponse; denoisedBlobUrl: string }> {
  const formData = new FormData();
  formData.append("file", file);

  // Use XMLHttpRequest for upload progress
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/denoise`);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.responseType = "blob";

    xhr.onload = async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        // The backend returns a JSON response with the audio URL,
        // or it may return the audio blob directly.
        const contentType = xhr.getResponseHeader("Content-Type") || "";

        if (contentType.includes("application/json")) {
          // JSON response — contains URL to download audio
          const text = await (xhr.response as Blob).text();
          const parsed = JSON.parse(text) as DenoiseResponse;
          const headerMetrics = parseProxyMetricsFromHeaders(xhr);
          let data: DenoiseResponse = {
            ...parsed,
            before_metrics: parsed.before_metrics ?? headerMetrics.before_metrics,
            after_metrics: parsed.after_metrics ?? headerMetrics.after_metrics,
          };
          // Fetch the actual audio file
          const audioRes = await fetch(`${API_BASE}${data.denoised_audio_url}`);
          const audioBlob = await audioRes.blob();

          if (!responseHasAnyMetrics(data)) {
            const computedMetrics = await computeProxyMetricsFromAudio(
              file,
              audioBlob,
            );
            if (computedMetrics) {
              data = {
                ...data,
                before_metrics:
                  data.before_metrics ?? computedMetrics.before_metrics,
                after_metrics: data.after_metrics ?? computedMetrics.after_metrics,
              };
            }
          }

          const denoisedBlobUrl = URL.createObjectURL(audioBlob);
          resolve({ data, denoisedBlobUrl });
        } else {
          // Direct audio blob response
          const blob = xhr.response as Blob;
          const denoisedBlobUrl = URL.createObjectURL(blob);

          // Extract metadata from headers if present
          const processingTime = xhr.getResponseHeader("X-Processing-Time");
          const headerMetrics = parseProxyMetricsFromHeaders(xhr);
          let data: DenoiseResponse = {
            denoised_audio_url: denoisedBlobUrl,
            original_filename: file.name,
            processing_time: processingTime
              ? parseFloat(processingTime)
              : undefined,
            ...headerMetrics,
          };

          if (!responseHasAnyMetrics(data)) {
            const computedMetrics = await computeProxyMetricsFromAudio(file, blob);
            if (computedMetrics) {
              data = {
                ...data,
                ...computedMetrics,
              };
            }
          }

          resolve({ data, denoisedBlobUrl });
        }
      } else {
        let msg = `Server error (${xhr.status})`;
        try {
          const text = await (xhr.response as Blob).text();
          const errJson = JSON.parse(text);
          msg = errJson.detail || errJson.message || msg;
        } catch {
          /* ignore parse errors */
        }
        reject(new ApiError(msg, xhr.status));
      }
    };

    xhr.onerror = () =>
      reject(new ApiError("Network error — is the backend running?", 0));
    xhr.ontimeout = () => reject(new ApiError("Request timed out", 0));
    xhr.timeout = REQUEST_TIMEOUT_MS;

    xhr.send(formData);
  });
}
