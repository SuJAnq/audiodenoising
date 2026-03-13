import { useEffect, useRef } from "react";

/**
 * Draws a waveform visualization with gradient fill from an audio URL.
 */
export default function WaveformVisualizer({ audioUrl }: { audioUrl: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!audioUrl) return;
    let cancelled = false;

    const draw = async () => {
      try {
        const res = await fetch(audioUrl);
        const arrayBuffer = await res.arrayBuffer();
        const audioCtx = new AudioContext();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        if (cancelled) {
          await audioCtx.close();
          return;
        }

        const channelData = audioBuffer.getChannelData(0);
        const canvas = canvasRef.current;
        if (!canvas) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const barWidth = 2;
        const barGap = 1;
        const totalBars = Math.floor(w / (barWidth + barGap));
        const step = Math.ceil(channelData.length / totalBars);
        const mid = h / 2;

        // Build gradient using CSS accent color
        const accent =
          getComputedStyle(document.documentElement)
            .getPropertyValue("--accent")
            .trim() || "#b8ff00";

        const gradient = ctx.createLinearGradient(0, 0, 0, h);
        gradient.addColorStop(0, accent);
        gradient.addColorStop(0.5, accent);
        gradient.addColorStop(1, "transparent");

        ctx.clearRect(0, 0, w, h);

        for (let i = 0; i < totalBars; i++) {
          let min = 1.0;
          let max = -1.0;
          const start = i * step;
          for (let j = 0; j < step; j++) {
            const val = channelData[start + j] ?? 0;
            if (val < min) min = val;
            if (val > max) max = val;
          }
          const amplitude = (max - min) / 2;
          const barH = Math.max(2, amplitude * h * 0.85);
          const x = i * (barWidth + barGap);
          const y = mid - barH / 2;

          ctx.globalAlpha = 0.5 + amplitude * 0.5;
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.roundRect(x, y, barWidth, barH, 1);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
        await audioCtx.close();
      } catch (e) {
        console.warn("Waveform render failed:", e);
      }
    };

    draw();
    return () => {
      cancelled = true;
    };
  }, [audioUrl]);

  return <canvas ref={canvasRef} className="waveform" />;
}
