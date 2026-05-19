"use client";
/**
 * CameraFeed — Dual Source + Auto-Record on Violence
 * ====================================================
 * WEBCAM:
 *   - Display:    Backend MJPEG stream (weapon boxes from Python)
 *   - Gender AI:  Hidden raw <video> → face-api (100% accuracy, same as original)
 *   - Recording:  MediaRecorder on raw webcam stream
 *                 Auto-starts when violence_detected = true
 *                 Saves 15s clip → pushed to Evidence with real video
 */

import { useEffect, useRef, useState, memo, useCallback } from "react";
import { CameraEntry } from "@/hooks/useCameraStore";
import { EvidenceEntry } from "@/hooks/useEvidenceStore";

const RECORDING_DURATION_MS = 15000; // record 15s per violence event
const GENDER_INTERVAL_MS    = 600;   // gender inference every 600ms
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8765";
const MJPEG_URL   = `${BACKEND_URL}/api/stream/mjpeg`;

interface Props {
  camera: CameraEntry;
  onRemove: (id: string) => void;
  onAlert?: (msg: string) => void;
  onGenderUpdate?: (male: number, female: number) => void;
  onDetection?: (entry: Omit<EvidenceEntry, "id">) => void;
  backendDetection?: {
    weapon_detected: boolean;
    weapon_label: string;
    weapon_confidence: number;
    violence_detected: boolean;
    violence_label: string;
    violence_confidence: number;
    threat_level?: string;
    motion_intensity?: number;
  } | null;
}

const CameraFeed = memo(function CameraFeed({
  camera, onRemove, onAlert, onGenderUpdate, onDetection, backendDetection,
}: Props) {

  // ── Refs ──────────────────────────────────────────────────────────────────
  const displayImgRef     = useRef<HTMLImageElement>(null);
  const displayImgCctvRef = useRef<HTMLImageElement>(null);
  const hiddenVideoRef    = useRef<HTMLVideoElement>(null);
  const canvasRef         = useRef<HTMLCanvasElement>(null);
  const faceapiRef        = useRef<any>(null);
  const streamRef         = useRef<MediaStream | null>(null);
  const rafRef            = useRef<number>(0);
  const lastRunRef        = useRef<number>(0);
  const modelsLoadedRef   = useRef(false);

  // Recording refs
  const mediaRecorderRef    = useRef<MediaRecorder | null>(null);
  const recordingChunksRef  = useRef<Blob[]>([]);
  const isRecordingRef      = useRef(false);
  const recordingTimerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const countIntervalRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const alarmAudioRef       = useRef<HTMLAudioElement | null>(null);
  const prevViolenceRef     = useRef(false);
  const detectionSnapshotRef = useRef<typeof backendDetection | null>(null);

  // ── State ─────────────────────────────────────────────────────────────────
  const [streamError, setStreamError] = useState(false);
  const [isLoaded, setIsLoaded]       = useState(false);
  const [modelsReady, setModelsReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recSeconds, setRecSeconds]   = useState(0);
  const [streamSrc, setStreamSrc]     = useState("");

  useEffect(() => {
    if (!camera) return;
    const base = camera.type === "cctv" ? camera.url : MJPEG_URL;
    if (!base) return;
    const separator = base.includes("?") ? "&" : "?";
    const latStr = camera.lat !== undefined ? `&lat=${camera.lat}` : "";
    const lngStr = camera.lng !== undefined ? `&lng=${camera.lng}` : "";
    const labelStr = camera.label ? `&label=${encodeURIComponent(camera.label)}` : "";
    setStreamSrc(`${base}${separator}camId=${camera.id}&t=${Date.now()}${latStr}${lngStr}${labelStr}`);
  }, [camera.id, camera.type, camera.url, camera.lat, camera.lng, camera.label]);

  const isWeaponAlert   = backendDetection?.weapon_detected   ?? false;
  const isViolenceAlert = backendDetection?.violence_detected ?? false;
  const isAnyAlert      = isWeaponAlert || isViolenceAlert;

  // Auto-play client-side looping alarm
  useEffect(() => {
    if (isViolenceAlert) {
      if (!alarmAudioRef.current) {
        alarmAudioRef.current = new Audio("/sound/alarm.wav");
        alarmAudioRef.current.loop = true;
      }
      alarmAudioRef.current.play().catch(e => console.log("[CameraFeed] Audio play delayed until gesture:", e));
    } else {
      if (alarmAudioRef.current) {
        alarmAudioRef.current.pause();
        alarmAudioRef.current.currentTime = 0;
      }
    }
    return () => {
      if (alarmAudioRef.current) {
        alarmAudioRef.current.pause();
        alarmAudioRef.current.currentTime = 0;
      }
    };
  }, [isViolenceAlert]);

  // ── 1. Load face-api models ───────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    async function loadModels() {
      try {
        const faceapi = await import("@vladmandic/face-api");
        if (!mounted) return;
        faceapiRef.current = faceapi;
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
          faceapi.nets.ageGenderNet.loadFromUri("/models"),
        ]);
        modelsLoadedRef.current = true;
        setModelsReady(true);
      } catch (err) {
        console.warn("[CameraFeed] Model load failed:", err);
      }
    }
    loadModels();
    return () => { mounted = false; };
  }, []);

  // ── 2. Hidden webcam for face-api + recording ─────────────────────────────
  useEffect(() => {
    if (camera.type !== "webcam") return;
    let stream: MediaStream | null = null;

    async function startHiddenWebcam() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false,
        });
        streamRef.current = stream;
        if (hiddenVideoRef.current) {
          hiddenVideoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.warn("[CameraFeed] Webcam access failed:", err);
      }
    }
    startHiddenWebcam();

    return () => {
      stream?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    };
  }, [camera.type]);

  // ── 2b. Dynamically select source on AI backend ────────────────────────────
  useEffect(() => {
    if (camera.type === "webcam" || camera.type === "cctv") {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8765";
      fetch(`${backendUrl}/api/select-source?source=0`, { method: "POST" }).catch(() => {});
    }
  }, [camera.type]);

  // ── 3. Auto-Record Logic ──────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (recordingTimerRef.current != null) clearTimeout(recordingTimerRef.current);
    if (countIntervalRef.current != null) clearInterval(countIntervalRef.current);
    isRecordingRef.current = false;
    setIsRecording(false);
    setRecSeconds(0);
  }, []);

  const startRecording = useCallback(() => {
    if (camera.type !== "webcam") return;
    if (isRecordingRef.current) return;
    const stream = streamRef.current;
    if (!stream) return;

    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : MediaRecorder.isTypeSupported("video/webm")
      ? "video/webm"
      : "";

    try {
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      mediaRecorderRef.current = mr;
      recordingChunksRef.current = [];
      isRecordingRef.current = true;
      setIsRecording(true);
      setRecSeconds(0);

      let elapsed = 0;
      countIntervalRef.current = setInterval(() => {
        elapsed++;
        setRecSeconds(elapsed);
        if (elapsed >= RECORDING_DURATION_MS / 1000 && countIntervalRef.current != null)
          clearInterval(countIntervalRef.current);
      }, 1000);

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) recordingChunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        if (countIntervalRef.current != null) clearInterval(countIntervalRef.current);
        const blob = new Blob(recordingChunksRef.current, { type: mimeType || "video/webm" });
        const videoUrl = URL.createObjectURL(blob);
        
        // Push to Evidence
        const snap = detectionSnapshotRef.current;
        onDetection?.({
          cameraId: camera.id,
          cameraLabel: camera.label,
          type: "VIOLENCE",
          confidence: snap?.violence_confidence ?? 0.9,
          timestamp: new Date().toLocaleTimeString(),
          isoTime: new Date().toISOString(),
          status: "Active",
          videoUrl,
          weaponDetected: snap?.weapon_detected ?? false,
          weaponType: snap?.weapon_label || undefined,
        });

        onAlert?.(`🎥 Violence recorded at ${camera.label}`);
        setIsRecording(false);
        isRecordingRef.current = false;
        
        // Optional: Revoke after some time if not used by Evidence Vault
        // setTimeout(() => URL.revokeObjectURL(videoUrl), 60000);
      };

      mr.start(1000);
      recordingTimerRef.current = setTimeout(() => {
        if (mr.state === "recording") mr.stop();
      }, RECORDING_DURATION_MS);

    } catch (err) {
      console.error("[CameraFeed] MediaRecorder failed:", err);
      setIsRecording(false);
      isRecordingRef.current = false;
    }
  }, [camera.id, camera.label, camera.type, onDetection, onAlert]);

  useEffect(() => {
    const violenceNow = backendDetection?.violence_detected ?? false;
    detectionSnapshotRef.current = backendDetection ?? null;
    if (violenceNow && !prevViolenceRef.current) {
      startRecording();
    }
    prevViolenceRef.current = violenceNow;
  }, [backendDetection, startRecording]);

  useEffect(() => {
    return () => {
      stopRecording();
      cancelAnimationFrame(rafRef.current);
    };
  }, [stopRecording]);

  // Keep dynamic ref of backend detection payload to use inside detectLoop without rebuilding
  const backendDetectionRef = useRef(backendDetection);
  useEffect(() => {
    backendDetectionRef.current = backendDetection;
  }, [backendDetection]);

  // ── 4. Gender AI Loop ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!modelsReady) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let stopped = false;

    // Capture non-null refs for use inside the async closure
    const c = canvas;
    const cx = ctx;

    async function detectLoop() {
      if (stopped) return;
      const now = Date.now();

      const detState = backendDetectionRef.current;
      const isHighMotion = isAnyAlert || 
        detState?.threat_level === "HIGH" || 
        detState?.threat_level === "CRITICAL" || 
        detState?.violence_detected ||
        (detState?.motion_intensity ?? 0) > 0.55;

      // During active violence/high threat scenes, skip face analysis completely to optimize CPU resources
      if (isHighMotion) {
        cx.clearRect(0, 0, c.width, c.height); // clear face boxes quickly to avoid ghosting
        rafRef.current = requestAnimationFrame(detectLoop);
        return;
      }

      // Reduce frequency under moderate threat scenes (2000ms instead of 600ms)
      const currentInterval = detState?.threat_level === "MEDIUM" ? 2000 : GENDER_INTERVAL_MS;

      if (now - lastRunRef.current >= currentInterval) {
        const source = camera.type === "webcam"
          ? hiddenVideoRef.current
          : camera.type === "upload"
          ? displayImgRef.current
          : displayImgCctvRef.current;
        const isReady = source && faceapiRef.current && modelsLoadedRef.current &&
          (source instanceof HTMLVideoElement ? source.readyState >= 2 : (source as HTMLImageElement).complete);

        if (isReady && source) {
          const { width, height } = source instanceof HTMLVideoElement
            ? { width: source.videoWidth, height: source.videoHeight }
            : { width: (source as HTMLImageElement).naturalWidth, height: (source as HTMLImageElement).naturalHeight };

          if (c.width !== width) {
            c.width = width;
            c.height = height;
          }
          cx.clearRect(0, 0, c.width, c.height);

          try {
            const detections = await faceapiRef.current
              .detectAllFaces(source, new faceapiRef.current.SsdMobilenetv1Options({ minConfidence: 0.4 }))
              .withAgeAndGender();

            onGenderUpdate?.(
              detections.filter((d: any) => d.gender === "male").length,
              detections.filter((d: any) => d.gender === "female").length
            );

            detections.forEach((det: any) => {
              const { x, y, width: boxW, height: boxH } = det.detection.box;
              const color = det.gender === "male" ? "#00aaff" : "#ff00aa";
              cx.strokeStyle = color; cx.lineWidth = 3;
              cx.strokeRect(x, y, boxW, boxH);
              cx.fillStyle = color;
              cx.font = "bold 16px Orbitron";
              cx.fillText(`${det.gender.toUpperCase()} ${Math.round(det.genderProbability * 100)}%`, x, y - 10);
            });
          } catch { /* non-fatal */ }
          lastRunRef.current = now;
        }
      }
      rafRef.current = requestAnimationFrame(detectLoop);
    }
    detectLoop();
    return () => { stopped = true; cancelAnimationFrame(rafRef.current); };
  }, [modelsReady, camera.type, onGenderUpdate, isAnyAlert]);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      position: "relative",
      overflow: "hidden",
      width: "100%",
      height: "100%",
      background: "#000",
      borderRadius: 4,
      border: isAnyAlert ? "2px solid var(--danger)" : "1px solid var(--border)",
      boxSizing: "border-box"
    }}>
      {camera.type === "webcam" && <video ref={hiddenVideoRef} autoPlay playsInline muted style={{ display: "none" }} />}
      
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        {streamSrc && !streamError && (
          <img 
            ref={camera.type === "cctv" ? displayImgCctvRef : displayImgRef} 
            src={streamSrc} 
            crossOrigin="anonymous" 
            alt="Live Stream" 
            style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} 
            onError={() => setStreamError(true)} 
            onLoad={() => setIsLoaded(true)} 
          />
        )}

        {streamError && (
          <div style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            background: "#18181b",
            color: "#71717a",
            zIndex: 5
          }}>
            <span style={{ fontSize: 24, marginBottom: 8 }}>📡</span>
            <span style={{ fontSize: 10, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: 1 }}>STREAM OFFLINE</span>
          </div>
        )}

        <canvas ref={canvasRef} style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          zIndex: 10
        }} />

        {/* HUD Elements */}
        <div style={{
          position: "absolute",
          bottom: 8,
          left: 8,
          zIndex: 20,
          display: "flex",
          alignItems: "center",
          gap: 8,
          background: "rgba(0,0,0,0.65)",
          padding: "4px 10px",
          borderRadius: 4,
          border: "1px solid rgba(255,255,255,0.12)",
          backdropFilter: "blur(4px)"
        }}>
          <div style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: isAnyAlert ? "var(--danger)" : "var(--safe)",
            animation: isAnyAlert ? "pulse 1s infinite alternate" : "none"
          }} />
          <span style={{ fontSize: 9, color: "#fff", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: 2 }}>{camera.label}</span>
        </div>

        {isRecording && (
          <div style={{
            position: "absolute",
            top: 8,
            left: 8,
            zIndex: 30,
            background: "rgba(220,38,38,0.9)",
            color: "#fff",
            fontSize: 9,
            padding: "4px 8px",
            borderRadius: 4,
            fontWeight: "bold",
            fontFamily: "monospace",
            letterSpacing: 1
          }}>
            REC {recSeconds}s
          </div>
        )}

        {isAnyAlert && (
          <div style={{
            position: "absolute",
            inset: 0,
            border: "3px solid rgba(220,38,38,0.85)",
            boxShadow: "inset 0 0 30px rgba(220,38,38,0.6)",
            animation: "pulseGlow 1.2s infinite alternate",
            pointerEvents: "none",
            zIndex: 15
          }} />
        )}

        <style>{`
          @keyframes pulseGlow {
            from { box-shadow: inset 0 0 15px rgba(220,38,38,0.4); border-color: rgba(220,38,38,0.6); }
            to { box-shadow: inset 0 0 35px rgba(220,38,38,0.95); border-color: rgba(220,38,38,1); }
          }
        `}</style>

        {isAnyAlert && (
          <div style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            zIndex: 20,
            background: "rgba(220,38,38,0.9)",
            color: "#fff",
            fontSize: 10,
            padding: "6px 0",
            textAlign: "center",
            fontWeight: "bold",
            fontFamily: "monospace",
            letterSpacing: 2,
            textTransform: "uppercase"
          }}>
            ⚠ {isWeaponAlert ? "WEAPON DETECTED" : `VIOLENCE DETECTED (${Math.round((backendDetection?.violence_confidence || 0.90) * 100)}%)`}
          </div>
        )}

        <button onClick={() => onRemove(camera.id)} style={{
          position: "absolute",
          top: 8,
          right: 8,
          zIndex: 30,
          width: 24,
          height: 24,
          background: "rgba(239,68,68,0.2)",
          border: "none",
          color: "#ef4444",
          fontSize: 16,
          fontWeight: "bold",
          borderRadius: 4,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          transition: "0.2s"
        }}
        onMouseEnter={e => { (e.target as any).style.background = "rgba(239,68,68,0.4)"; }}
        onMouseLeave={e => { (e.target as any).style.background = "rgba(239,68,68,0.2)"; }}>×</button>
      </div>
    </div>
  );
});

export default CameraFeed;
