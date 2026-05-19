"use client";
import { motion } from "framer-motion";
import { EvidenceEntry } from "@/hooks/useEvidenceStore";

interface Props {
  evidence: EvidenceEntry;
  onClose: () => void;
  onUpdate: (id: string, updates: Partial<EvidenceEntry>) => void;
}

export default function EvidenceDetail({ evidence, onClose, onUpdate }: Props) {
  const mono: React.CSSProperties = { fontFamily: "monospace" };

  const handleDispatch = () => {
    let stationName = "Mumbai Headquarters";
    try {
      const authStr = localStorage.getItem("sd_auth");
      if (authStr) {
        stationName = JSON.parse(authStr).user || stationName;
      }
    } catch {}

    onUpdate(evidence.id, {
      status: "Police Dispatched",
      authorityStation: stationName,
      dispatchTime: new Date().toLocaleTimeString()
    });
  };

  const handleResolve = () => {
    onUpdate(evidence.id, { status: "Resolved" });
  };

  const handleMoreHelp = () => {
    let stationName = "Mumbai Headquarters";
    try {
      const authStr = localStorage.getItem("sd_auth");
      if (authStr) {
        stationName = JSON.parse(authStr).user || stationName;
      }
    } catch {}

    onUpdate(evidence.id, {
      status: "More Help Requested",
      authorityStation: stationName
    });
  };

  const googleMapsUrl = `https://maps.google.com/?q=${evidence.lat},${evidence.lng}`;

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      style={{ position: "fixed", inset: 0, background: "rgba(5,8,16,0.95)", zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
        style={{ background: "var(--bg2)", border: "1px solid var(--border)", width: "100%", maxWidth: 900, maxHeight: "90vh", overflowY: "auto", position: "relative" }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ padding: "15px 20px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between", background: "var(--bg1)" }}>
          <div>
            <div style={{ ...mono, fontSize: 10, color: "var(--accent)", letterSpacing: 2 }}>INCIDENT DETAILS</div>
            <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 18, fontWeight: 800, color: "var(--text-bright)" }}>{evidence.id}</div>
          </div>
          <button onClick={onClose} style={{ background: "transparent", border: "none", color: "var(--text-dim)", fontSize: 24, cursor: "pointer" }}>×</button>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 0 }}>
          {/* Left Side: Visuals */}
          <div style={{ borderRight: "1px solid var(--border)" }}>
            <div style={{ background: "#000", aspectRatio: "16/9", width: "100%", height: 350, position: "relative" }}>
              {evidence.videoUrl ? (
                evidence.videoUrl.includes("/stream/mjpeg") ? (
                  <img src={evidence.videoUrl} style={{ width: "100%", height: "100%", objectFit: "contain" }} alt="Live Surveillance Feed" />
                ) : (
                  <video src={evidence.videoUrl} controls autoPlay loop style={{ width: "100%", height: "100%" }} />
                )
              ) : (
                <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "var(--bg3)" }}>
                  <div style={{ fontSize: 40, marginBottom: 10 }}>📹</div>
                  <div style={{ ...mono, fontSize: 12, color: "var(--text-dim)" }}>VIDEO BUFFER UNAVAILABLE</div>
                </div>
              )}
            </div>

            <div style={{ padding: 20, display: "flex", gap: 10 }}>
              <a
                href={evidence.videoUrl}
                download={`Evidence_${evidence.id}.mp4`}
                style={{ flex: 1, textDecoration: "none", textAlign: "center", ...mono, fontSize: 12, background: "var(--accent)", color: "#000", padding: "12px", fontWeight: 700, cursor: "pointer" }}
              >
                📥 DOWNLOAD MP4 RECORDING
              </a>
            </div>
          </div>

          {/* Right Side: Data */}
          <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Status and Threat Level Badges */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ padding: "10px", background: (evidence.status || "Active") === "Active" ? "rgba(255,34,68,0.15)" : "rgba(0,255,170,0.15)", border: `1px solid ${(evidence.status || "Active") === "Active" ? "var(--danger)" : "var(--accent)"}`, color: (evidence.status || "Active") === "Active" ? "var(--danger)" : "var(--accent)", ...mono, fontSize: 11, textAlign: "center", fontWeight: 700, letterSpacing: 1 }}>
                STATUS: {(evidence.status || "Active").toUpperCase()}
              </div>

              {/* Dynamic Threat Level Badge */}
              <div style={{
                padding: "10px",
                background: evidence.type?.includes("CRITICAL") || evidence.type?.includes("Fighting") ? "rgba(255,0,0,0.2)" : evidence.type?.includes("HIGH") ? "rgba(255,100,0,0.15)" : "rgba(0,255,170,0.1)",
                border: `1px solid ${evidence.type?.includes("CRITICAL") || evidence.type?.includes("Fighting") ? "#ff3333" : evidence.type?.includes("HIGH") ? "#ff8800" : "var(--accent)"}`,
                color: evidence.type?.includes("CRITICAL") || evidence.type?.includes("Fighting") ? "#ff3333" : evidence.type?.includes("HIGH") ? "#ffaa00" : "var(--accent)",
                ...mono,
                fontSize: 11,
                textAlign: "center",
                fontWeight: 700,
                letterSpacing: 1
              }}>
                LEVEL: {evidence.type?.toUpperCase().replace("[", "").replace("]", "") || "TACTICAL"}
              </div>
            </div>

            {/* Incident Metadata */}
            <div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)", marginBottom: 5 }}>INCIDENT METADATA</div>
              <div style={{ background: "var(--bg3)", padding: 15, border: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ ...mono, fontSize: 12, color: "var(--text-bright)", display: "flex", justifyContent: "space-between" }}>
                  <span>THREAT CLASSIFICATION:</span>
                  <span style={{ color: "var(--danger)", fontWeight: 700 }}>{evidence.type || "VIOLENCE"}</span>
                </div>
                <div style={{ ...mono, fontSize: 12, color: "var(--text-bright)", display: "flex", justifyContent: "space-between" }}>
                  <span>AGGRESSION SCORE:</span>
                  <span style={{ color: "var(--warning)", fontWeight: 700 }}>{(evidence.confidence * 100).toFixed(0)}%</span>
                </div>
                <div style={{ ...mono, fontSize: 12, color: "var(--text-bright)", display: "flex", justifyContent: "space-between" }}>
                  <span>RECORDING DURATION:</span>
                  <span style={{ color: "var(--accent)", fontWeight: 700 }}>{evidence.duration || "15 sec"}</span>
                </div>
                <div style={{ ...mono, fontSize: 12, color: "var(--text-bright)", display: "flex", justifyContent: "space-between" }}>
                  <span>CAPTURED TIME:</span>
                  <span style={{ color: "var(--text)", fontWeight: 700 }}>{evidence.timestamp}</span>
                </div>
                {evidence.isoTime && (
                  <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)", borderTop: "1px solid var(--border)", paddingTop: 8, marginTop: 4 }}>
                    OCCURRED: {new Date(evidence.isoTime).toLocaleString("en-IN")}
                  </div>
                )}
              </div>
            </div>

            {/* Location */}
            <div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)", marginBottom: 5 }}>LOCATION METADATA</div>
              <div style={{ background: "var(--bg3)", padding: 15, border: "1px solid var(--border)" }}>
                <div style={{ ...mono, fontSize: 13, color: "var(--text-bright)", marginBottom: 4 }}>SOURCE: {evidence.cameraLabel} <span style={{ color: "var(--text-dim)", fontSize: 10 }}>({evidence.cameraId})</span></div>
                <div style={{ ...mono, fontSize: 11, color: "var(--text-dim)" }}>LAT: {evidence.lat?.toFixed(6)}</div>
                <div style={{ ...mono, fontSize: 11, color: "var(--text-dim)", marginBottom: 12 }}>LNG: {evidence.lng?.toFixed(6)}</div>

                <a href={googleMapsUrl} target="_blank" rel="noreferrer" style={{ display: "flex", alignItems: "center", gap: 8, ...mono, fontSize: 11, color: "var(--accent)", textDecoration: "none", border: "1px solid var(--accent)", padding: "6px 12px", width: "fit-content" }}>
                  📍 VIEW ON GOOGLE MAPS
                </a>
              </div>
            </div>

            {/* AI Analysis */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ background: "var(--bg3)", padding: 12 }}>
                <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", marginBottom: 4 }}>GENDER ANALYSIS</div>
                <div style={{ ...mono, fontSize: 12, color: "var(--text)" }}>Male: {evidence.maleCount || 0}</div>
                <div style={{ ...mono, fontSize: 12, color: "var(--text)" }}>Female: {evidence.femaleCount || 0}</div>
              </div>
              <div style={{ background: "var(--bg3)", padding: 12 }}>
                <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", marginBottom: 4 }}>WEAPON DETECTION</div>
                <div style={{ ...mono, fontSize: 12, color: evidence.weaponDetected ? "var(--danger)" : "var(--text)" }}>
                  {evidence.weaponDetected ? `DETECTED (${evidence.weaponType || "Knife"})` : "None"}
                </div>
              </div>
            </div>

            {/* Responding Authority details */}
            {evidence.authorityStation && (
              <div style={{ background: "rgba(0,170,255,0.08)", border: "1px solid rgba(0,170,255,0.25)", padding: 12, ...mono, fontSize: 11 }}>
                <div style={{ color: "var(--accent)", fontWeight: 700, marginBottom: 4 }}>👮 RESPONDING AUTHORITY</div>
                <div style={{ color: "var(--text-bright)" }}>STATION: {evidence.authorityStation}</div>
                {evidence.dispatchTime && <div style={{ color: "var(--text-dim)", fontSize: 10, marginTop: 2 }}>DISPATCHED AT: {evidence.dispatchTime}</div>}
              </div>
            )}

            {/* Authority Action Section */}
            <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
              {evidence.status === "Active" && (
                <button onClick={handleDispatch} style={{ background: "var(--accent)", color: "#000", border: "none", padding: "14px", fontWeight: 800, cursor: "pointer", ...mono }}>
                  🚀 DISPATCH EMERGENCY RESPONSE
                </button>
              )}
              {(evidence.status === "Police Dispatched" || evidence.status === "More Help Requested") && (
                <div style={{ display: "flex", gap: 8 }}>
                  {evidence.status !== "More Help Requested" && (
                    <button onClick={handleMoreHelp} style={{ flex: 1, background: "var(--warning)", color: "#000", border: "none", padding: "12px", fontWeight: 700, cursor: "pointer", ...mono }}>
                      🆘 REQUEST MORE HELP
                    </button>
                  )}
                  <button onClick={handleResolve} style={{ flex: 1, background: "var(--accent)", color: "#000", border: "none", padding: "12px", fontWeight: 700, cursor: "pointer", ...mono }}>
                    ✅ MARK AS RESOLVED
                  </button>
                </div>
              )}
              {evidence.status === "Resolved" && (
                <div style={{ textAlign: "center", ...mono, fontSize: 11, color: "var(--accent)", padding: 10, background: "rgba(0,255,170,0.05)" }}>
                  ✓ THIS INCIDENT HAS BEEN RESOLVED
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
