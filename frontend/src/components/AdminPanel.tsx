"use client";
import { useState, useEffect } from "react";
import { CameraEntry } from "@/hooks/useCameraStore";
import { usePoliceStore } from "@/hooks/usePoliceStore";

interface Props {
  cameras: CameraEntry[];
  onRemoveCamera: (id: string) => void;
  currentUser: string;
}

interface BackendHealth {
  status: string;
  version: string;
  uptime_s: number;
  cameras_active: boolean;
  weapon_detector: boolean;
  violence_detector: boolean;
  human_detector: boolean;
  pose_estimator: boolean;
}

interface ModelStatus {
  pipeline_fps: number;
  persons_in_frame: number;
  weapon_avg_inference_ms: number;
  startup_time: string;
}

export default function AdminPanel({ cameras, onRemoveCamera, currentUser }: Props) {
  const { accounts, addAccount, removeAccount } = usePoliceStore();
  const [newName, setNewName] = useState("");
  const [newLoc, setNewLoc] = useState("");
  const [newPass, setNewPass] = useState("");

  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [evidenceCount, setEvidenceCount] = useState(0);

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  // Fetch real backend metrics (Section 5)
  useEffect(() => {
    async function fetchAdminStats() {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8765";
      try {
        const resHealth = await fetch(`${backendUrl}/api/health`);
        if (resHealth.ok) {
          const data = await resHealth.json();
          setHealth(data);
        }
        
        const resModels = await fetch(`${backendUrl}/api/model-status`);
        if (resModels.ok) {
          const data = await resModels.json();
          setModelStatus(data);
        }

        const resEvidence = await fetch(`${backendUrl}/api/evidence`);
        if (resEvidence.ok) {
          const data = await resEvidence.json();
          setEvidenceCount(data.length);
        }
      } catch (err) {
        console.warn("Failed to fetch admin backend stats:", err);
      }
    }

    fetchAdminStats();
    const interval = setInterval(fetchAdminStats, 2000);
    return () => clearInterval(interval);
  }, []);

  const browserInfo = typeof navigator !== "undefined" ? navigator.userAgent.split(" ").slice(-2).join(" ") : "Unknown";
  const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
  const camCount = cameras.length;
  const onlineCount = cameras.filter(c => c.status === "active").length;

  // Format uptime_s
  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600).toString().padStart(2, "0");
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
    const s = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${h}:${m}:${s}`;
  };

  const uptimeStr = health ? formatUptime(health.uptime_s) : "00:00:00";

  return (
    <div style={{ display: "flex", flexDirection: "column", padding: 20, gap: 16, overflowY: "auto", height: "100%", background: "var(--bg1)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 16, fontWeight: 700, letterSpacing: 4, color: "var(--accent)" }}>SYSTEM ADMIN</div>
        <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)", marginLeft: "auto" }}>OPERATOR: {currentUser}</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Police Station Services */}
        <AdminCard title="AUTHORIZED RESPONSE PLAZAS" style={{ gridColumn: "1 / -1" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 20 }}>
            <div style={{ borderRight: "1px solid var(--border)", paddingRight: 20 }}>
              <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 12 }}>Active dispatch channels linked with real-time WebSocket state routing.</div>
              {accounts.length === 0 ? (
                 <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)" }}>NO STATIONS CREATED</div>
              ) : (
                accounts.map((acc, idx) => (
                  <div key={acc.id || acc.name || idx} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid rgba(26,40,64,0.3)" }}>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{acc.name}</div>
                      <div style={{ ...mono, fontSize: 8, color: "var(--accent2)" }}>{acc.location} · PASS: ••••••••</div>
                    </div>
                    <button onClick={() => removeAccount(acc.id)} style={{ background: "transparent", border: "1px solid var(--danger)", color: "var(--danger)", ...mono, fontSize: 8, padding: "2px 6px", cursor: "pointer" }}>REVOKE</button>
                  </div>
                ))
              )}
            </div>
            
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
               <div style={{ ...mono, fontSize: 10, color: "var(--accent2)", letterSpacing: 1 }}>PROVISION RESPONSE CENTER</div>
               <input placeholder="STATION NAME" value={newName} onChange={e => setNewName(e.target.value)}
                 style={{ background: "var(--bg3)", border: "1px solid var(--border)", padding: 8, color: "var(--text)", ...mono, fontSize: 11 }} />
               <input placeholder="LOCATION AREA" value={newLoc} onChange={e => setNewLoc(e.target.value)}
                 style={{ background: "var(--bg3)", border: "1px solid var(--border)", padding: 8, color: "var(--text)", ...mono, fontSize: 11 }} />
               <input placeholder="PASSWORD" type="password" value={newPass} onChange={e => setNewPass(e.target.value)}
                 style={{ background: "var(--bg3)", border: "1px solid var(--border)", padding: 8, color: "var(--text)", ...mono, fontSize: 11 }} />
               <button onClick={() => { addAccount(newName, newLoc, newPass); setNewName(""); setNewLoc(""); setNewPass(""); }}
                 style={{ background: "var(--accent)", color: "#000", border: "none", padding: 10, ...mono, fontSize: 10, fontWeight: 700, cursor: "pointer" }}>
                 + PROVISION ACCESS
               </button>
            </div>
          </div>
        </AdminCard>

        {/* AI Engine Status */}
        <AdminCard title="AI PIPELINE ENGINE STATUS">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>HUMAN DETECTOR</div>
              <div style={{ ...mono, fontSize: 10, color: health?.human_detector ? "var(--safe)" : "var(--danger)", fontWeight: 700 }}>
                {health?.human_detector ? "ACTIVE (YOLOv8n ONNX)" : "INACTIVE / LOADING"}
              </div>
            </div>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>POSE ESTIMATION AGENT</div>
              <div style={{ ...mono, fontSize: 10, color: health?.pose_estimator ? "var(--safe)" : "var(--danger)", fontWeight: 700 }}>
                {health?.pose_estimator ? "ACTIVE (MediaPipe Lite Task)" : "INACTIVE / LOADING"}
              </div>
            </div>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>VIOLENCE CLASSIFIER</div>
              <div style={{ ...mono, fontSize: 10, color: health?.violence_detector ? "var(--safe)" : "var(--danger)", fontWeight: 700 }}>
                {health?.violence_detector ? "ACTIVE (Temporal ONNX LSTM)" : "INACTIVE / LOADING"}
              </div>
            </div>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>WEAPON DETECTOR</div>
              <div style={{ ...mono, fontSize: 10, color: health?.weapon_detector ? "var(--safe)" : "var(--text-dim)", fontWeight: 700 }}>
                {health?.weapon_detector ? "ACTIVE (ONNX Custom)" : "DISABLED"}
              </div>
            </div>
          </div>
        </AdminCard>

        {/* Operational Statistics */}
        <AdminCard title="OPERATIONAL TELEMETRY">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>PIPELINE RENDERING RATE</div>
              <div style={{ ...mono, fontSize: 11, color: "var(--accent)", fontWeight: 700 }}>
                {modelStatus ? `${modelStatus.pipeline_fps.toFixed(1)} FPS` : "0.0 FPS"}
              </div>
            </div>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>WEAPON INFERENCE SPEED</div>
              <div style={{ ...mono, fontSize: 11, color: "var(--text)", fontWeight: 700 }}>
                {modelStatus && modelStatus.weapon_avg_inference_ms > 0 ? `${modelStatus.weapon_avg_inference_ms.toFixed(1)} ms` : "N/A"}
              </div>
            </div>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>EVIDENCE RECORDINGS VAULT</div>
              <div style={{ ...mono, fontSize: 11, color: "var(--text)", fontWeight: 700 }}>
                {evidenceCount} Persistent JSON entries
              </div>
            </div>
          </div>
        </AdminCard>

        {/* System Info */}
        <AdminCard title="TACTICAL HARDWARE & SECURITY INFO" style={{ gridColumn: "1 / -1" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20 }}>
            {[
              ["SurakshaDrishti core", health ? `v${health.version}` : "v4.0.0"],
              ["Uptime Uptime", uptimeStr],
              ["Active Cameras", `${onlineCount} / ${camCount}`],
              ["Connection Protocols", isHttps ? "Secure (HTTPS)" : "Standard (HTTP/WS)"],
              ["Local Vault Path", "backend/recordings/"],
              ["Environment Context", browserInfo],
            ].map(([l, v]) => (
              <div key={l}>
                <div style={{ fontSize: 9, color: "var(--text-dim)", textTransform: "uppercase" }}>{l}</div>
                <div style={{ ...mono, fontSize: 10, color: "var(--text-bright)", fontWeight: 600 }}>{v}</div>
              </div>
            ))}
          </div>
        </AdminCard>

      </div>
    </div>
  );
}

function AdminCard({ title, children, style }: { title: string; children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ background: "var(--panel)", border: "1px solid var(--border)", padding: 16, ...style }}>
      <div style={{ fontFamily: "monospace", fontSize: 9, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 12, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        {title}
      </div>
      {children}
    </div>
  );
}
