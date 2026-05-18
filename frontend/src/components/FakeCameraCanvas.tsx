"use client";
import { useEffect, useRef } from "react";

interface Props {
  alertMode?: boolean;
  width?: number;
  height?: number;
}

export default function FakeCameraCanvas({ alertMode = false, width = 400, height = 240 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef  = useRef(0);
  const rafRef    = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    function draw() {
      const w = canvas!.width, h = canvas!.height;
      ctx.fillStyle = alertMode ? "#0a0508" : "#05080d";
      ctx.fillRect(0, 0, w, h);

      // Noise
      for (let i = 0; i < 120; i++) {
        ctx.fillStyle = `rgba(255,255,255,${Math.random() * 0.04})`;
        ctx.fillRect(Math.random() * w, Math.random() * h, 1, 1);
      }

      if (alertMode) {
        ctx.fillStyle = "rgba(255,34,68,0.1)";
        ctx.fillRect(0, 0, w, h);
        // Figures
        ctx.fillStyle = "rgba(200,80,80,0.6)";
        ctx.fillRect(160, 80, 20, 60);
        ctx.beginPath(); ctx.arc(170, 72, 10, 0, Math.PI * 2); ctx.fill();
        ctx.fillRect(190, 85, 18, 55);
        ctx.beginPath(); ctx.arc(199, 78, 9, 0, Math.PI * 2); ctx.fill();
        // Detection box
        const pulse = 0.5 + 0.5 * Math.sin(frameRef.current * 0.1);
        ctx.strokeStyle = `rgba(255,34,68,${0.6 + pulse * 0.4})`;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(148, 62, 68, 82);
        // Label
        ctx.fillStyle = `rgba(255,34,68,${0.8 + pulse * 0.2})`;
        ctx.fillRect(148, 54, 68, 14);
        ctx.fillStyle = "#fff";
        ctx.font = "8px monospace";
        ctx.fillText("VIOLENCE 91%", 152, 64);
      } else {
        ctx.fillStyle = "rgba(0,170,255,0.04)";
        ctx.fillRect(0, 0, w, h);
        const x = 80 + (frameRef.current * 1.2 % (w - 100));
        ctx.fillStyle = "rgba(100,160,200,0.5)";
        ctx.fillRect(x, 100, 14, 50);
        ctx.beginPath(); ctx.arc(x + 7, 93, 8, 0, Math.PI * 2); ctx.fill();
      }

      // Scanlines
      for (let y = 0; y < h; y += 4) {
        ctx.fillStyle = "rgba(0,0,0,0.12)";
        ctx.fillRect(0, y, w, 2);
      }

      frameRef.current++;
      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [alertMode]);

  return <canvas ref={canvasRef} width={width} height={height} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />;
}
