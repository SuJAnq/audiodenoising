import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AudioUploader from "../components/AudioUploader";
import WaveformVisualizer from "../components/WaveformVisualizer";
import SpectrogramVisualizer from "../components/SpectrogramVisualizer";
import { denoiseAudio } from "../services/audioApi";
import type { DenoiseResponse, ProcessingState } from "../types/audio";

export default function HomePage({
  onDenoiseComplete,
}: {
  onDenoiseComplete: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<ProcessingState>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DenoiseResponse | null>(null);
  const [denoisedUrl, setDenoisedUrl] = useState<string | null>(null);
  const [isExtraInfoVisible, setIsExtraInfoVisible] = useState(true);

  const originalUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  );

  const prevUrls = useRef<string[]>([]);
  const trackUrl = (url: string) => {
    prevUrls.current.push(url);
  };

  const handleFileSelect = useCallback((f: File) => {
    setResult(null);
    setDenoisedUrl(null);
    setIsExtraInfoVisible(true);
    setError(null);
    setProgress(0);
    setState("idle");
    setFile(f);
  }, []);

  const handleRemove = useCallback(() => {
    setFile(null);
    setResult(null);
    setDenoisedUrl(null);
    setIsExtraInfoVisible(true);
    setState("idle");
    setError(null);
  }, []);

  const handleDenoise = useCallback(async () => {
    if (!file || state === "uploading" || state === "processing") return;
    setState("uploading");
    setProgress(0);
    setError(null);

    try {
      const { data, denoisedBlobUrl } = await denoiseAudio(file, (pct) => {
        setProgress(pct);
        if (pct >= 100) setState("processing");
      });
      trackUrl(denoisedBlobUrl);
      setResult(data);
      setDenoisedUrl(denoisedBlobUrl);
      setState("done");
      onDenoiseComplete();
    } catch (e: any) {
      setError(e.message ?? "Something went wrong.");
      setState("error");
    }
  }, [file, state, onDenoiseComplete]);

  // One-step UX: once a file is selected, start denoising automatically.
  useEffect(() => {
    if (file && state === "idle") {
      void handleDenoise();
    }
  }, [file, state, handleDenoise]);

  /* Auto-scroll to results page after denoising completes */
  useEffect(() => {
    if (state === "done") {
      setIsExtraInfoVisible(true);
      setTimeout(() => {
        document
          .getElementById("page-results")
          ?.scrollIntoView({ behavior: "smooth" });
      }, 400);
    }
  }, [state]);

  const isProcessing = state === "uploading" || state === "processing";

  return (
    <>
      {/* ═══════════ PAGE 1 — Upload ═══════════ */}
      <section id="page-upload" className="fullpage__section">
        <div className="fullpage__content fullpage__content--center">
          <div className="hero">
            <span className="hero__badge">
              <span className="hero__badge-dot" />
              Powered by Deep Learning
            </span>
            <h1 className="hero__title">
              <span className="hero__title-accent">AI-Powered</span>
              <br />
              Audio Denoiser
            </h1>
            <p className="hero__subtitle">
              Remove background noise from your audio recordings using a U-Net
              deep learning model. Upload a file and get a clean version in
              seconds.
            </p>
          </div>

          <div className="upload-section">
            <div className="upload-card">
              <AudioUploader
                file={file}
                onFileSelect={handleFileSelect}
                onRemove={handleRemove}
                disabled={isProcessing}
              />

              {file && state === "idle" && (
                <div className="status status--success">
                  File selected. Starting denoising...
                </div>
              )}

              {isProcessing && (
                <div className="processing">
                  <div className="processing__spinner" />
                  <div className="processing__text">
                    {state === "uploading"
                      ? `Uploading… ${progress}%`
                      : "Processing with AI model — this may take a moment…"}
                  </div>
                  <div className="processing__bar">
                    <div className="processing__bar-fill" />
                  </div>
                </div>
              )}

              {state === "error" && error && (
                <>
                  <div className="status status--error">⚠️ {error}</div>
                  <button
                    className="btn btn--secondary"
                    onClick={handleDenoise}
                  >
                    Retry
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Scroll-down hint arrow */}
        <div className="scroll-hint" aria-hidden="true">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </section>

      {/* ═══════════ PAGE 2 — Results (only after denoising) ═══════════ */}
      {state === "done" && denoisedUrl && (
        <section id="page-results" className="fullpage__section">
          <div className="fullpage__content fullpage__content--scroll">
            <div className="results-page">
              <div className="results-page__header">
                <div>
                  <div className="section-label">Output</div>
                  <h2 className="section-title" style={{ marginBottom: 0 }}>
                    Denoised Result
                  </h2>
                </div>
                <span className="result-card__badge">Status: Done</span>
              </div>

              {result?.processing_time != null && (
                <div
                  className="status status--success"
                  style={{ marginBottom: "1rem" }}
                >
                  ✅ Processed in {result.processing_time.toFixed(2)}s
                </div>
              )}

              <button
                type="button"
                className="results-toggle"
                aria-expanded={isExtraInfoVisible}
                aria-controls="results-extra-info"
                onClick={() => setIsExtraInfoVisible((visible) => !visible)}
              >
                <span>
                  {isExtraInfoVisible ? "Hide metrics" : "Show metrics"}
                </span>
                <svg
                  className="results-toggle__icon"
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </button>

              <div
                id="results-extra-info"
                className={[
                  "results-extra-info",
                  isExtraInfoVisible ? "results-extra-info--visible" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                aria-hidden={!isExtraInfoVisible}
              >
                {result?.metrics && (
                  <div className="metrics">
                    {result.metrics.snr != null && (
                      <div className="metric">
                        <div className="metric__value">
                          {result.metrics.snr.toFixed(2)}
                        </div>
                        <div className="metric__label">SNR (dB)</div>
                      </div>
                    )}
                    {result.metrics.psnr != null && (
                      <div className="metric">
                        <div className="metric__value">
                          {result.metrics.psnr.toFixed(2)}
                        </div>
                        <div className="metric__label">PSNR</div>
                      </div>
                    )}
                    {result.metrics.ssim != null && (
                      <div className="metric">
                        <div className="metric__value">
                          {result.metrics.ssim.toFixed(4)}
                        </div>
                        <div className="metric__label">SSIM</div>
                      </div>
                    )}
                    {result.metrics.lsd != null && (
                      <div className="metric">
                        <div className="metric__value">
                          {result.metrics.lsd.toFixed(3)}
                        </div>
                        <div className="metric__label">LSD</div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="results-grid">
                <div className="results-column">
                  <div className="audio-player__label">Original (Noisy)</div>
                  {originalUrl && <SpectrogramVisualizer audioUrl={originalUrl} />}
                  {originalUrl && <WaveformVisualizer audioUrl={originalUrl} />}
                  {originalUrl && (
                    <audio controls src={originalUrl} preload="metadata" />
                  )}
                </div>
                <div className="results-column">
                  <div className="audio-player__label">Denoised (Clean)</div>
                  <SpectrogramVisualizer audioUrl={denoisedUrl} />
                  <WaveformVisualizer audioUrl={denoisedUrl} />
                  <audio controls src={denoisedUrl} preload="metadata" />
                </div>
              </div>

              {/* Actions */}
              <div className="results-actions">
                <button className="btn btn--secondary" onClick={handleRemove}>
                  Denoise Another
                </button>
                <a
                  className="btn btn--accent"
                  href={denoisedUrl}
                  download={`denoised_${file?.name ?? "audio.wav"}`}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  Download
                </a>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ═══════════ PAGE 3 — How It Works ═══════════ */}
      <section
        id="page-how"
        className="fullpage__section fullpage__section--between"
      >
        <div className="fullpage__content fullpage__content--center">
          <div className="section-label">Process</div>
          <h2 className="section-title">How It Works</h2>
          <div className="steps">
            <div className="step">
              <span className="step__number">1</span>
              <div className="step__content">
                <h3>Upload</h3>
                <p>Select or drag a noisy audio file (WAV, MP3, FLAC, etc.).</p>
              </div>
            </div>
            <div className="step">
              <span className="step__number">2</span>
              <div className="step__content">
                <h3>Process</h3>
                <p>
                  Our U-Net model analyzes the spectrogram and separates noise
                  from signal.
                </p>
              </div>
            </div>
            <div className="step">
              <span className="step__number">3</span>
              <div className="step__content">
                <h3>Download</h3>
                <p>
                  Listen to the comparison and download your clean audio file.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer — only visible on this last page */}
        <footer className="footer">
          <div className="footer__links">
            <a href="/blog" className="footer__link">
              Blog
            </a>
            <span className="footer__dot">&middot;</span>
            <a
              href="https://github.com"
              className="footer__link"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </div>
          <div className="footer__copy">
            <span
              style={{ fontFamily: "var(--font-display)", fontWeight: 600 }}
            >
              DenoiseAI
            </span>{" "}
            &mdash; U-Net &middot; FastAPI &middot; {new Date().getFullYear()}
          </div>
        </footer>
      </section>
    </>
  );
}
