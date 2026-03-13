/* === Types for the audio denoising frontend === */

export interface DenoiseResponse {
  /** URL/path to the denoised audio file served by the backend */
  denoised_audio_url: string;
  /** Original filename */
  original_filename: string;
  /** Optional metrics returned by the model */
  metrics?: DenoiseMetrics;
  /** Processing time in seconds */
  processing_time?: number;
}

export interface DenoiseMetrics {
  snr?: number;
  psnr?: number;
  ssim?: number;
  lsd?: number;
}

export interface HealthStatus {
  status: "ok" | "error";
  model_loaded: boolean;
  version?: string;
}

export type ProcessingState =
  | "idle"
  | "uploading"
  | "processing"
  | "done"
  | "error";
