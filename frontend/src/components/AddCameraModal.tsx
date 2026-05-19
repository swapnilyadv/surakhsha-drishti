"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  onAdd: (cam: {
    type: "cctv" | "webcam" | "upload";
    label: string;
    url?: string;
    lat?: number;
    lng?: number;
    status: "active";
  }) => void;
  onClose: () => void;
}

export default function AddCameraModal({ onAdd, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<"cctv" | "webcam" | "upload">("cctv");
  const [label, setLabel] = useState("");
  const [url, setUrl] = useState("");
  const [lat, setLat] = useState("");
  const [lng, setLng] = useState("");
  const [gpsLoading, setGpsLoading] = useState(false);
  const [ipLoading, setIpLoading] = useState(false);
  const [error, setError] = useState("");

  // File upload state variables
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  function getDeviceGPS() {
    setGpsLoading(true);
    setError("");
    navigator.geolocation.getCurrentPosition(
      pos => {
        setLat(pos.coords.latitude.toFixed(6));
        setLng(pos.coords.longitude.toFixed(6));
        setGpsLoading(false);
      },
      () => { setError("GPS access denied — please enter manually"); setGpsLoading(false); },
      { timeout: 8000 }
    );
  }

  async function getIPLocation() {
    setIpLoading(true);
    setError("");
    try {
      const res = await fetch("https://ip-api.com/json/?fields=lat,lon,city,status");
      const data = await res.json();
      if (data.status === "success") {
        setLat(data.lat.toFixed(6));
        setLng(data.lon.toFixed(6));
      } else {
        setError("IP location failed — enter manually");
      }
    } catch { setError("IP location failed — enter manually"); }
    setIpLoading(false);
  }

  async function submit() {
    if (!label.trim() && activeTab !== "upload") { setError("Label is required"); return; }
    
    if (activeTab === "cctv") {
      if (!url.trim()) { setError("Stream URL is required for CCTV"); return; }
      if (!lat || !lng) { setError("Location is required for CCTV"); return; }
    }

    // Resolve dynamic uploader location inheritance (Task 7)
    let finalLat = lat ? parseFloat(lat) : undefined;
    let finalLng = lng ? parseFloat(lng) : undefined;

    if (!finalLat || !finalLng) {
      try {
        const res = await fetch("https://ip-api.com/json/?fields=lat,lon,status");
        const data = await res.json();
        if (data.status === "success") {
          finalLat = parseFloat(data.lat.toFixed(6));
          finalLng = parseFloat(data.lon.toFixed(6));
        }
      } catch {}
      if (!finalLat || !finalLng) {
        finalLat = 19.0760;
        finalLng = 72.8777;
      }
    }

    if (activeTab === "upload") {
      if (!selectedFile) { setError("Please select a video file to upload"); return; }
      
      setIsUploading(true);
      setError("");
      setUploadProgress(0);

      const formData = new FormData();
      formData.append("file", selectedFile);

      try {
        const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8765";
        const xhr = new XMLHttpRequest();
        
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            setUploadProgress(percent);
          }
        };

        xhr.onload = () => {
          setIsUploading(false);
          if (xhr.status >= 200 && xhr.status < 300) {
            const finalLabel = label.trim() || selectedFile.name.replace(/\.[^/.]+$/, "");
            onAdd({
              type: "upload",
              label: finalLabel,
              status: "active",
              lat: finalLat,
              lng: finalLng,
            });
            onClose();
          } else {
            try {
              const res = JSON.parse(xhr.responseText);
              setError(res.message || "Failed to upload video");
            } catch {
              setError("Failed to upload video");
            }
          }
        };

        xhr.onerror = () => {
          setIsUploading(false);
          setError("Connection to backend server failed");
        };

        xhr.open("POST", `${backendUrl}/api/upload-video`);
        xhr.send(formData);
        return;
      } catch (err) {
        setIsUploading(false);
        setError("Failed to initialize video upload");
        return;
      }
    }

    onAdd({
      type: activeTab,
      label: label.trim(),
      url: activeTab === "cctv" ? url.trim() : undefined,
      status: "active",
      lat: finalLat,
      lng: finalLng,
    });
    onClose();
  }

  function populateMJPEGUrl() {
    const host = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
      ? '127.0.0.1' 
      : window.location.hostname;
    setUrl(`http://${host}:8765/api/stream/mjpeg`);
  }

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.85)" }}
      onClick={e => e.target === e.currentTarget && onClose()}>
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}
        style={{ width: 500, background: "var(--bg2)", border: "1px solid var(--border-glow)", padding: 28, position: "relative", maxHeight: "90vh", overflowY: "auto", boxShadow: "0 0 50px rgba(0,0,0,0.5)" }}>

         <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 22 }}>
          <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 14, fontWeight: 700, letterSpacing: 3, color: "var(--accent)" }}>
            📡 CONNECT SOURCE
          </div>
          <button onClick={onClose} style={{ background: "transparent", border: "none", color: "var(--text-dim)", fontSize: 24, cursor: "pointer", lineHeight: 1 }}>×</button>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 2, background: "rgba(255,255,255,0.05)", padding: 2, marginBottom: 20 }}>
          {(["cctv", "webcam", "upload"] as const).map(t => (
            <button key={t} onClick={() => setActiveTab(t)}
              style={{ flex: 1, padding: "10px", ...mono, fontSize: 10, letterSpacing: 2, border: "none", cursor: "pointer", background: activeTab === t ? "var(--accent2)" : "transparent", color: activeTab === t ? "#fff" : "var(--text-dim)", transition: "0.2s" }}>
              {t.toUpperCase()} {t === "upload" ? "VIDEO" : "SOURCE"}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <Field label="Identification Label" value={label} onChange={setLabel} placeholder={activeTab === "cctv" ? "e.g. Platform CCTV-01" : activeTab === "webcam" ? "e.g. Mobile Unit Alpha" : "e.g. Uploaded Clip Alpha"} />
          
          {activeTab === "cctv" && (
            <div>
              <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase", marginBottom: 5 }}>Stream URL (MJPEG / HLS)</div>
              <div style={{ display: "flex", gap: 8 }}>
                <input value={url} onChange={e => setUrl(e.target.value)} placeholder="http://192.168.1.100:8080/video"
                  style={{ flex: 1, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 13, padding: "9px 12px", outline: "none" }}/>
                <button onClick={populateMJPEGUrl} style={{ ...mono, fontSize: 10, background: "rgba(0,150,100,0.15)", border: "1px solid rgba(0,200,150,0.4)", color: "var(--safe)", padding: "8px 12px", cursor: "pointer", letterSpacing: 1 }}>
                  AUTO
                </button>
              </div>
            </div>
          )}

          {activeTab === "webcam" && (
            <div style={{ ...mono, fontSize: 10, color: "var(--accent)", background: "rgba(0,170,255,0.05)", border: "1px solid rgba(0,170,255,0.2)", padding: 12, lineHeight: 1.6 }}>
              ✓ Browser MediaStream will be used.<br/>
              ✓ Real-time AI processing enabled locally.<br/>
              ⚠ Requires camera permissions.
            </div>
          )}

          {activeTab === "upload" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase" }}>Select Video File</div>
              <input type="file" accept=".mp4,.mov,.avi,.mkv" disabled={isUploading} onChange={e => {
                if (e.target.files && e.target.files[0]) {
                  setSelectedFile(e.target.files[0]);
                  if (!label) {
                    setLabel(e.target.files[0].name.replace(/\.[^/.]+$/, ""));
                  }
                }
              }}
                style={{ width: "100%", background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "9px 12px", cursor: isUploading ? "not-allowed" : "pointer" }} />
              
              {isUploading && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", ...mono, fontSize: 10, marginBottom: 4 }}>
                    <span style={{ color: "var(--accent)" }}>UPLOADING...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div style={{ height: 4, background: "rgba(255,255,255,0.1)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${uploadProgress}%`, background: "var(--accent2)", transition: "width 0.2s" }} />
                  </div>
                </div>
              )}

              <div style={{ ...mono, fontSize: 10, color: "var(--accent)", background: "rgba(0,170,255,0.05)", border: "1px solid rgba(0,170,255,0.2)", padding: 12, lineHeight: 1.6, marginTop: 4 }}>
                ✓ Supported formats: MP4, MOV, AVI, MKV.<br/>
                ✓ Runs through full real-time human, pose, violence, and weapon detection pipelines.<br/>
                ✓ Automatically streams overlays directly to the dashboard.
              </div>
            </div>
          )}

          <div>
            <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>Geolocation Bind</div>
            <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <input value={lat} onChange={e => setLat(e.target.value)} placeholder="Lat"
                style={{ flex: 1, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "8px 10px", outline: "none" }}/>
              <input value={lng} onChange={e => setLng(e.target.value)} placeholder="Lng"
                style={{ flex: 1, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "8px 10px", outline: "none" }}/>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={getIPLocation} disabled={ipLoading}
                style={{ flex: 1, ...mono, fontSize: 9, background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", padding: "7px", cursor: "pointer" }}>
                {ipLoading ? "..." : "IP LOCATE"}
              </button>
              <button onClick={getDeviceGPS} disabled={gpsLoading}
                style={{ flex: 1, ...mono, fontSize: 9, background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", padding: "7px", cursor: "pointer" }}>
                {gpsLoading ? "..." : "DEVICE GPS"}
              </button>
            </div>
          </div>

          {error && <div style={{ ...mono, fontSize: 10, color: "var(--danger)", textAlign: "center" }}>⚠ {error}</div>}
          
          <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
            <button onClick={onClose} disabled={isUploading} style={{ flex: 1, padding: "12px", ...mono, fontSize: 11, background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", cursor: isUploading ? "not-allowed" : "pointer" }}>CANCEL</button>
            <button onClick={submit} disabled={isUploading} style={{ flex: 2, padding: "12px", fontFamily: "Orbitron,sans-serif", fontSize: 12, fontWeight: 700, background: isUploading ? "rgba(255,255,255,0.1)" : "var(--accent2)", border: "none", color: isUploading ? "var(--text-dim)" : "#fff", cursor: isUploading ? "not-allowed" : "pointer", letterSpacing: 2 }}>
              {isUploading ? "UPLOADING..." : `INITIALIZE ${activeTab.toUpperCase()}`}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder: string }) {
  const mono: React.CSSProperties = { fontFamily: "monospace" };
  return (
    <div>
      <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase", marginBottom: 5 }}>{label}</div>
      <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        style={{ width: "100%", background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 13, padding: "9px 12px", outline: "none" }}/>
    </div>
  );
}