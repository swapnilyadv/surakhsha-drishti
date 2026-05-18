"use client";
import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { EvidenceEntry } from "@/hooks/useEvidenceStore";
import EvidenceDetail from "./EvidenceDetail";

interface Props {
  evidence: EvidenceEntry[];
  onClearAll: () => void;
  onUpdate: (id: string, updates: Partial<EvidenceEntry>) => void;
}

export default function EvidenceVault({ evidence, onClearAll, onUpdate }: Props) {
  const [selected, setSelected] = useState<EvidenceEntry | null>(null);
  const mono: React.CSSProperties = { fontFamily: "monospace" };

  function confirmClear() {
    if (window.confirm("Clear all evidence? This frees memory and cannot be undone. Evidence will not reappear after reload.")) {
      onClearAll();
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Detail View Modal */}
      <AnimatePresence>
        {selected && (
          <EvidenceDetail 
            evidence={selected} 
            onClose={() => setSelected(null)} 
            onUpdate={(id, up) => {
              onUpdate(id, up);
              setSelected(prev => prev ? { ...prev, ...up } : null);
            }}
          />
        )}
      </AnimatePresence>

      {/* Header */}
      <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border)", background: "var(--bg2)", display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
        <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 15, fontWeight: 700, letterSpacing: 4, color: "var(--accent)" }}>
          EVIDENCE VAULT
        </div>
        <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)" }}>{evidence.length} RECORD{evidence.length !== 1 ? "S" : ""}</div>

        {evidence.length > 0 && (
          <button onClick={confirmClear}
            style={{ marginLeft: "auto", ...mono, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", background: "rgba(255,34,68,0.08)", border: "1px solid rgba(255,34,68,0.4)", color: "var(--danger)", padding: "6px 14px", cursor: "pointer" }}
            onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,34,68,0.2)"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,34,68,0.08)"; }}>
            🗑 CLEAR ALL EVIDENCE
          </button>
        )}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: "auto", padding: evidence.length > 0 ? 16 : 0 }}>
        {evidence.length === 0 ? (
          /* Empty state */
          <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, opacity: 0.5 }}>
            <svg width="72" height="72" viewBox="0 0 72 72" fill="none">
              <circle cx="36" cy="36" r="30" stroke="#4a6080" strokeWidth="1.5"/>
              <path d="M24 48V30l12-10 12 10v18" stroke="#4a6080" strokeWidth="1.5" strokeLinejoin="round"/>
              <rect x="28" y="40" width="6" height="8" rx="1" stroke="#4a6080" strokeWidth="1.2"/>
              <rect x="38" y="34" width="6" height="6" rx="1" stroke="#4a6080" strokeWidth="1.2"/>
            </svg>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 15, fontWeight: 700, color: "var(--text-dim)", letterSpacing: 4, marginBottom: 8 }}>
                NO EVIDENCE RECORDED
              </div>
              <div style={{ ...mono, fontSize: 11, color: "var(--text-dim)", lineHeight: 1.7 }}>
                Evidence is automatically captured when<br/>
                violence or harassment is detected through<br/>
                your connected cameras.
              </div>
            </div>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(360px,1fr))", gap: 14 }}>
            <AnimatePresence>
              {evidence.map((ev, i) => (
                <motion.div key={ev.id}
                  initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ delay: i * 0.04 }}
                  onClick={() => setSelected(ev)}
                  style={{ background: "var(--panel)", border: `1px solid var(--border)`, borderLeft: `3px solid ${ev.type.includes("VIOLENCE") || ev.type.includes("HARASSMENT") ? "var(--danger)" : "var(--warning)"}`, overflow: "hidden", cursor: "pointer" }}
                  whileHover={{ y: -2, borderColor: "#00ffaa" }}>

                  {/* Thumbnail */}
                  <div style={{ height: 160, background: "var(--bg3)", position: "relative", overflow: "hidden" }}>
                    {ev.thumbnail ? (
                      <img src={ev.thumbnail} alt="Detection frame" style={{ width: "100%", height: "100%", objectFit: "cover" }}/>
                    ) : (
                      <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 6, opacity: 0.4 }}>
                        <div style={{ fontSize: 24 }}>📹</div>
                        <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)" }}>FRAME UNAVAILABLE AFTER RELOAD</div>
                      </div>
                    )}
                    {/* Status Badge Overlay */}
                    <div style={{ position: "absolute", top: 8, right: 8, ...mono, fontSize: 8, background: (ev.status || "Active") === "Active" ? "rgba(255,34,68,0.8)" : "rgba(0,255,170,0.8)", color: "#000", padding: "3px 8px", fontWeight: 700 }}>
                      {(ev.status || "Active").toUpperCase()}
                    </div>
                    {/* Type badge */}
                    <div style={{ position: "absolute", top: 8, left: 8, ...mono, fontSize: 9, background: "rgba(5,8,16,0.85)", color: ev.type.includes("VIOLENCE") ? "var(--danger)" : "var(--warning)", padding: "3px 8px", border: `1px solid ${ev.type.includes("VIOLENCE") ? "rgba(255,34,68,0.5)" : "rgba(255,170,0,0.5)"}`, letterSpacing: 1 }}>
                      {ev.type}
                    </div>
                  </div>

                  <div style={{ padding: 14 }}>
                    <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 1, marginBottom: 6 }}>{ev.id}</div>
                    <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 14, fontWeight: 700, color: "var(--text-bright)", marginBottom: 8 }}>
                      {ev.cameraLabel}
                    </div>

                    {/* Meta */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5, marginBottom: 10 }}>
                      {[
                        ["Male", ev.maleCount ?? 0],
                        ["Female", ev.femaleCount ?? 0],
                        ["Weapon", ev.weaponDetected ? (ev.weaponType || "Knife") : "None"],
                        ["Location", ev.locationName?.split(" ")[0] || "Unknown"],
                        ["Detected", ev.timestamp]
                      ].map(([l, v]) => (
                        <div key={l} style={{ background: "var(--bg3)", padding: "5px 8px", gridColumn: l === "Detected" ? "span 2" : "auto" }}>
                          <div style={{ ...mono, fontSize: 7, color: "var(--text-dim)", letterSpacing: 1, marginBottom: 2 }}>{l}</div>
                          <div style={{ ...mono, fontSize: 10, color: l === "Weapon" && v !== "None" ? "var(--danger)" : "var(--text)" }}>{v}</div>
                        </div>
                      ))}
                    </div>

                    <div style={{ ...mono, fontSize: 9, color: "var(--accent)", textAlign: "center", border: "1px solid rgba(0,255,170,0.2)", padding: 6 }}>
                      CLICK TO VIEW FULL DETAILS & RECORDING
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}
