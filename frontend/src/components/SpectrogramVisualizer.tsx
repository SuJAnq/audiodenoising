import { useEffect, useRef } from "react";

/* ── Cooley-Tukey radix-2 FFT (in-place) ── */
function fft(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wR = Math.cos(angle);
    const wI = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let cR = 1,
        cI = 0;
      for (let j = 0; j < half; j++) {
        const a = i + j,
          b = a + half;
        const tR = real[b] * cR - imag[b] * cI;
        const tI = real[b] * cI + imag[b] * cR;
        real[b] = real[a] - tR;
        imag[b] = imag[a] - tI;
        real[a] += tR;
        imag[a] += tI;
        const nR = cR * wR - cI * wI;
        cI = cR * wI + cI * wR;
        cR = nR;
      }
    }
  }
}

/* ── Inferno-inspired colormap ── */
const CMAP: [number, number, number][] = [
  [0, 0, 4],
  [27, 12, 65],
  [82, 20, 103],
  [137, 32, 106],
  [185, 57, 87],
  [222, 94, 59],
  [249, 142, 9],
  [252, 204, 37],
  [252, 255, 164],
];

function mapColor(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const idx = t * (CMAP.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, CMAP.length - 1);
  const f = idx - lo;
  return [
    Math.round(CMAP[lo][0] + (CMAP[hi][0] - CMAP[lo][0]) * f),
    Math.round(CMAP[lo][1] + (CMAP[hi][1] - CMAP[lo][1]) * f),
    Math.round(CMAP[lo][2] + (CMAP[hi][2] - CMAP[lo][2]) * f),
  ];
}

export default function SpectrogramVisualizer({
  audioUrl,
}: {
  audioUrl: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!audioUrl) return;
    let cancelled = false;

    const render = async () => {
      try {
        const res = await fetch(audioUrl);
        const buf = await res.arrayBuffer();
        const actx = new AudioContext();
        const abuf = await actx.decodeAudioData(buf);
        if (cancelled) {
          await actx.close();
          return;
        }

        const samples = abuf.getChannelData(0);
        const canvas = canvasRef.current;
        if (!canvas) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const cw = Math.round(rect.width * dpr);
        const ch = Math.round(rect.height * dpr);
        canvas.width = cw;
        canvas.height = ch;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const fftSize = 1024;
        const hop = 512;
        const bins = fftSize / 2;
        const frames = Math.max(
          1,
          Math.floor((samples.length - fftSize) / hop) + 1,
        );

        /* Compute STFT → dB magnitudes */
        const spec: Float32Array[] = [];
        let peak = -Infinity;

        for (let f = 0; f < frames; f++) {
          const off = f * hop;
          const re = new Float32Array(fftSize);
          const im = new Float32Array(fftSize);
          for (let i = 0; i < fftSize; i++) {
            const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
            re[i] = (samples[off + i] ?? 0) * w;
          }
          fft(re, im);
          const mag = new Float32Array(bins);
          for (let k = 0; k < bins; k++) {
            const m = Math.sqrt(re[k] ** 2 + im[k] ** 2);
            mag[k] = 20 * Math.log10(Math.max(m, 1e-10));
            if (mag[k] > peak) peak = mag[k];
          }
          spec.push(mag);
        }

        /* Render spectrogram to canvas */
        const range = 80; // dB dynamic range
        const floor = peak - range;
        const img = ctx.createImageData(cw, ch);

        for (let x = 0; x < cw; x++) {
          const fi = Math.min(Math.floor((x / cw) * frames), frames - 1);
          const mag = spec[fi];
          for (let y = 0; y < ch; y++) {
            const bi = Math.min(
              Math.floor(((ch - 1 - y) / ch) * bins),
              bins - 1,
            );
            const norm = Math.max(0, Math.min(1, (mag[bi] - floor) / range));
            const [r, g, b] = mapColor(norm);
            const p = (y * cw + x) * 4;
            img.data[p] = r;
            img.data[p + 1] = g;
            img.data[p + 2] = b;
            img.data[p + 3] = 255;
          }
        }

        ctx.putImageData(img, 0, 0);
        await actx.close();
      } catch (e) {
        console.warn("Spectrogram render failed:", e);
      }
    };

    render();
    return () => {
      cancelled = true;
    };
  }, [audioUrl]);

  return <canvas ref={canvasRef} className="spectrogram" />;
}
