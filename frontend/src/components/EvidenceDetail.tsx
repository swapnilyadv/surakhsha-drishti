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
    onUpdate(evidence.id, {
      status: "Police Dispatched",
      authorityStation: "Zone-7 Police Headquarter",
      dispatchTime: new Date().toLocaleTimeString()
    });
  };

  const handleResolve = () => {
    onUpdate(evidence.id, { status: "Resolved" });
  };

  const handleMoreHelp = () => {
    onUpdate(evidence.id, { status: "More Help Requested" });
  };

  const googleMapsUrl = `https://www.google.com/maps?q=${evidence.lat},${evidence.lng}`;

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
                <video src={evidence.videoUrl} controls autoPlay loop style={{ width: "100%", height: "100%" }} />
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
          <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Status Badge */}
            <div style={{ padding: "10px 15px", background: (evidence.status || "Active") === "Active" ? "rgba(255,34,68,0.15)" : "rgba(0,255,170,0.15)", border: `1px solid ${(evidence.status || "Active") === "Active" ? "var(--danger)" : "var(--accent)"}`, color: (evidence.status || "Active") === "Active" ? "var(--danger)" : "var(--accent)", ...mono, fontSize: 12, textAlign: "center", fontWeight: 700, letterSpacing: 2 }}>
              STATUS: {(evidence.status || "Active").toUpperCase()}
            </div>

            {/* Location */}
            <div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)", marginBottom: 5 }}>LOCATION METADATA</div>
              <div style={{ background: "var(--bg3)", padding: 15, border: "1px solid var(--border)" }}>
                <div style={{ ...mono, fontSize: 13, color: "var(--text-bright)", marginBottom: 4 }}>{evidence.locationName}</div>
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

            {/* Authority Action Section */}
            <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", gap: 8 }}>
              {evidence.status === "Active" && (
                <button onClick={handleDispatch} style={{ background: "var(--accent)", color: "#000", border: "none", padding: "14px", fontWeight: 800, cursor: "pointer", ...mono }}>
                  🚀 DISPATCH EMERGENCY RESPONSE
                </button>
              )}
              {evidence.status === "Police Dispatched" && (
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={handleMoreHelp} style={{ flex: 1, background: "var(--warning)", color: "#000", border: "none", padding: "12px", fontWeight: 700, cursor: "pointer", ...mono }}>
                    🆘 REQUEST MORE HELP
                  </button>
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
