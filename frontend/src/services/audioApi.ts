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
          const data: DenoiseResponse = {
            ...parsed,
            before_metrics: parsed.before_metrics ?? headerMetrics.before_metrics,
            after_metrics: parsed.after_metrics ?? headerMetrics.after_metrics,
          };
          // Fetch the actual audio file
          const audioRes = await fetch(`${API_BASE}${data.denoised_audio_url}`);
          const audioBlob = await audioRes.blob();
          const denoisedBlobUrl = URL.createObjectURL(audioBlob);
          resolve({ data, denoisedBlobUrl });
        } else {
          // Direct audio blob response
          const blob = xhr.response as Blob;
          const denoisedBlobUrl = URL.createObjectURL(blob);

          // Extract metadata from headers if present
          const processingTime = xhr.getResponseHeader("X-Processing-Time");
          const headerMetrics = parseProxyMetricsFromHeaders(xhr);
          const data: DenoiseResponse = {
            denoised_audio_url: denoisedBlobUrl,
            original_filename: file.name,
            processing_time: processingTime
              ? parseFloat(processingTime)
              : undefined,
            ...headerMetrics,
          };
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
