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
  const prevViolenceRef     = useRef(false);
  const detectionSnapshotRef = useRef<typeof backendDetection | null>(null);

  // ── State ─────────────────────────────────────────────────────────────────
  const [streamError, setStreamError] = useState(false);
  const [isLoaded, setIsLoaded]       = useState(false);
  const [modelsReady, setModelsReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recSeconds, setRecSeconds]   = useState(0);

  const isWeaponAlert   = backendDetection?.weapon_detected   ?? false;
  const isViolenceAlert = backendDetection?.violence_detected ?? false;
  const isAnyAlert      = isWeaponAlert || isViolenceAlert;

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

      if (now - lastRunRef.current >= GENDER_INTERVAL_MS) {
        const source = camera.type === "webcam" ? hiddenVideoRef.current : displayImgCctvRef.current;
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
  }, [modelsReady, camera.type, onGenderUpdate]);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="relative overflow-hidden h-full w-full bg-black rounded" style={{ border: isAnyAlert ? "2px solid var(--danger)" : "1px solid var(--border)" }}>
      {camera.type === "webcam" && <video ref={hiddenVideoRef} autoPlay playsInline muted className="hidden" />}
      
      <div className="relative w-full h-full">
        {camera.type === "webcam" ? (
          !streamError && <img ref={displayImgRef} src={MJPEG_URL} alt="Live" className="w-full h-full object-cover" onError={() => setStreamError(true)} onLoad={() => setIsLoaded(true)} />
        ) : (
          camera.url && !streamError && <img ref={displayImgCctvRef} src={camera.url} crossOrigin="anonymous" alt="CCTV" className="w-full h-full object-cover" onError={() => setStreamError(true)} onLoad={() => setIsLoaded(true)} />
        )}

        {streamError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-900 text-zinc-500">
            <span className="text-2xl">📡</span>
            <span className="text-[10px] mt-2 font-mono">STREAM OFFLINE</span>
          </div>
        )}

        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-10" />

        {/* HUD Elements */}
        <div className="absolute bottom-2 left-2 z-20 flex items-center gap-2 bg-black/60 px-2 py-1 rounded border border-white/10">
          <div className={`w-2 h-2 rounded-full ${isAnyAlert ? "bg-red-500 animate-pulse" : "bg-green-500"}`} />
          <span className="text-[9px] text-white font-mono uppercase tracking-widest">{camera.label}</span>
        </div>

        {isRecording && (
          <div className="absolute top-2 left-2 z-30 bg-red-600/90 text-white text-[9px] px-2 py-1 rounded font-bold animate-pulse">
            REC {recSeconds}s
          </div>
        )}

        {isAnyAlert && (
          <div className="absolute top-0 left-0 right-0 z-20 bg-red-600/80 text-white text-[10px] py-1 text-center font-bold tracking-tighter animate-blink">
            ⚠ {isWeaponAlert ? "WEAPON DETECTED" : "VIOLENCE DETECTED"}
          </div>
        )}

        <button onClick={() => onRemove(camera.id)} className="absolute top-2 right-2 z-30 w-6 h-6 bg-red-500/20 hover:bg-red-500/40 text-red-500 rounded flex items-center justify-center">×</button>
      </div>
    </div>
  );
});

export default CameraFeed;
