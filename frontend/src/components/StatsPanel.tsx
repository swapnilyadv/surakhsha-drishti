"use client";
import { useEffect, useState } from "react";
import { CameraEntry } from "@/hooks/useCameraStore";
import MiniMap from "./MiniMap";
import LiveIncidentCard from "./LiveIncidentCard";
import { EvidenceEntry } from "@/hooks/useEvidenceStore";
import { useBackendAI } from "@/hooks/useBackendAI";

interface Props {
  isAlert: boolean;
  evidence: EvidenceEntry[];
  cameraCount: number;
  activeCameraCount: number;
  cameras: CameraEntry[];
  alertCamIds: Set<string>;
  onUpdateEvidence: (id: string, updates: Partial<EvidenceEntry>) => void;
  liveMale?: number;
  liveFemale?: number;
}

export default function StatsPanel({
  isAlert, evidence, cameraCount, activeCameraCount,
  cameras, alertCamIds, onUpdateEvidence,
  liveMale = 0, liveFemale = 0,
}: Props) {
  const activeIncidents = evidence.slice(0, 10);
  const [sessionStart] = useState(Date.now());
  const [uptime, setUptime] = useState("00:00:00");
  const [memoryUsage, setMemoryUsage] = useState<string>("N/A");

  // Real-time backend detection data (violence %, weapon status)
  const { detection, connected } = useBackendAI();
  const violenceConf = detection.violence_detected ? Math.round(detection.violence_confidence * 100) : 0;
  const violenceDetected = detection.violence_detected;
  const weaponDetected = detection.weapon_detected;
  // Use backend or prop for alert state
  const isAnyAlert = isAlert || violenceDetected || weaponDetected;

  useEffect(() => {
    const timer = setInterval(() => {
      const diff = Date.now() - sessionStart;
      const h = Math.floor(diff / 3600000).toString().padStart(2, "0");
      const m = Math.floor((diff % 3600000) / 60000).toString().padStart(2, "0");
      const s = Math.floor((diff % 60000) / 1000).toString().padStart(2, "0");
      setUptime(`${h}:${m}:${s}`);
      if ((performance as any).memory) {
        const used = (performance as any).memory.usedJSHeapSize;
        setMemoryUsage(`${Math.round(used / 1048576)} MB`);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [sessionStart]);

  const mono: React.CSSProperties = { fontFamily: "monospace" };
  const totalPeople = liveMale + liveFemale;

  return (
    <div style={{ width: 300, flexShrink: 0, display: "flex", flexDirection: "column", overflowY: "auto", background: "var(--bg2)", borderLeft: "1px solid var(--border)" }}>

      {/* Mini Map */}
      <div style={{ borderBottom: "1px solid var(--border)", background: "#050810" }}>
        <div style={{ padding: "8px 14px", ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 1 }}>TACTICAL OVERVIEW</div>
        <div style={{ width: "100%", height: 160 }}>
          <MiniMap cameras={cameras} alertCamIds={alertCamIds} />
        </div>
      </div>

      {/* ── VIOLENCE PROBABILITY (compact, real data) ── */}
      <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)", background: "linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%)" }}>
        <div style={{ ...mono, fontSize: 8, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 4, height: 4, borderRadius: "50%", background: violenceDetected ? "var(--danger)" : "var(--safe)", flexShrink: 0, animation: violenceDetected ? "pulse-dot 0.8s infinite" : "none" }} />
          VIOLENCE PROBABILITY
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 10 }}>
          <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 26, fontWeight: 900, lineHeight: 1, color: violenceDetected ? "var(--danger)" : "var(--safe)", textShadow: violenceDetected ? "0 0 18px rgba(255,34,68,0.5)" : "0 0 18px rgba(0,255,136,0.3)" }}>
            {violenceConf.toFixed(0)}%
          </div>
          <div style={{ ...mono, fontSize: 9, color: violenceDetected ? "var(--danger)" : "var(--safe)", letterSpacing: 1 }}>
            {violenceDetected ? "HIGH RISK" : "NOMINAL"}
          </div>
        </div>
        <div style={{ height: 6, background: "rgba(0,0,0,0.4)", borderRadius: 3, overflow: "hidden", border: "1px solid rgba(255,255,255,0.04)" }}>
          <div style={{ height: "100%", width: `${Math.max(violenceConf, 0.5)}%`, background: violenceDetected ? "var(--danger)" : "var(--safe)", boxShadow: violenceDetected ? "0 0 10px var(--danger)" : "none", transition: "width 1s cubic-bezier(0.4,0,0.2,1)" }} />
        </div>
      </div>

      {/* ── SESSION STATUS (compact, real data) ── */}
      <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ ...mono, fontSize: 8, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 8 }}>SESSION STATUS</div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 26, fontWeight: 900, lineHeight: 1, color: isAnyAlert ? "var(--danger)" : "var(--safe)", textShadow: isAnyAlert ? "0 0 18px rgba(255,34,68,0.5)" : "0 0 18px rgba(0,255,136,0.3)" }}>
            {isAnyAlert ? "ALERT" : "STABLE"}
          </div>
          <span style={{ ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 1 }}>
            {violenceDetected ? "VIOLENCE ACTIVE" : weaponDetected ? "WEAPON DETECTED" : isAlert ? "THREAT ACTIVE" : "ENV. SECURE"}
          </span>
        </div>
        <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          {[
            ["SESSION UPTIME", uptime],
            ["CAMERAS", `${activeCameraCount} / ${cameraCount}`],
            ["JS MEMORY", memoryUsage],
            ["STORAGE", "LOCAL"],
          ].map(([l, v]) => (
            <div key={l} style={{ background: "rgba(255,255,255,0.02)", padding: "7px 10px", border: "1px solid var(--border)", borderRadius: 2 }}>
              <div style={{ ...mono, fontSize: 7, color: "var(--text-dim)", letterSpacing: 1, marginBottom: 4 }}>{l}</div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text)", fontWeight: 600 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── LIVE GENDER COUNT (NEW) ── */}
      <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ ...mono, fontSize: 8, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 4, height: 4, borderRadius: "50%", background: totalPeople > 0 ? "#00aaff" : "var(--text-dim)", animation: totalPeople > 0 ? "pulse-dot 1.5s infinite" : "none", flexShrink: 0 }} />
          LIVE POPULATION SCAN
        </div>

        {/* Male + Female big counters */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
          {/* Male */}
          <div style={{ background: "rgba(0,170,255,0.07)", border: "1px solid rgba(0,170,255,0.25)", borderRadius: 3, padding: "10px 12px", textAlign: "center" }}>
            <div style={{ ...mono, fontSize: 7, color: "#00aaff", letterSpacing: 2, marginBottom: 4 }}>MALE</div>
            <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 28, fontWeight: 900, color: "#00aaff", lineHeight: 1, textShadow: "0 0 12px rgba(0,170,255,0.4)" }}>
              {liveMale}
            </div>
            <div style={{ ...mono, fontSize: 7, color: "rgba(0,170,255,0.5)", marginTop: 3 }}>DETECTED</div>
          </div>

          {/* Female */}
          <div style={{ background: "rgba(255,0,170,0.07)", border: "1px solid rgba(255,0,170,0.25)", borderRadius: 3, padding: "10px 12px", textAlign: "center" }}>
            <div style={{ ...mono, fontSize: 7, color: "#ff00aa", letterSpacing: 2, marginBottom: 4 }}>FEMALE</div>
            <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 28, fontWeight: 900, color: "#ff00aa", lineHeight: 1, textShadow: "0 0 12px rgba(255,0,170,0.4)" }}>
              {liveFemale}
            </div>
            <div style={{ ...mono, fontSize: 7, color: "rgba(255,0,170,0.5)", marginTop: 3 }}>DETECTED</div>
          </div>
        </div>

        {/* Total bar */}
        <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)", borderRadius: 3, padding: "7px 10px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 1 }}>TOTAL PERSONS</div>
          <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 16, fontWeight: 700, color: totalPeople > 0 ? "var(--text)" : "var(--text-dim)" }}>
            {totalPeople}
          </div>
        </div>

        {/* Gender split bar */}
        {totalPeople > 0 && (
          <div style={{ marginTop: 8, height: 4, background: "rgba(255,255,255,0.05)", borderRadius: 2, overflow: "hidden", display: "flex" }}>
            <div style={{ width: `${(liveMale / totalPeople) * 100}%`, background: "#00aaff", transition: "width 0.8s ease" }} />
            <div style={{ width: `${(liveFemale / totalPeople) * 100}%`, background: "#ff00aa", transition: "width 0.8s ease" }} />
          </div>
        )}
        {totalPeople === 0 && (
          <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", textAlign: "center", marginTop: 6, letterSpacing: 1 }}>
            NO PERSONS IN FRAME
          </div>
        )}
      </div>

      {/* Live Incident Tracker */}
      <Block label="LIVE INCIDENT TRACKER">
        <div className="space-y-4 max-h-[400px] overflow-y-auto pr-1">
          {activeIncidents.length === 0 ? (
            <div className="py-8 text-center border border-dashed border-zinc-800">
              <div className="text-zinc-600 text-[10px] font-mono tracking-widest">SYSTEM SECURE</div>
              <div className="text-zinc-800 text-[8px] font-mono mt-1 uppercase">No active threats detected</div>
            </div>
          ) : (
            activeIncidents.map(inc => (
              <LiveIncidentCard
                key={inc.id}
                incident={inc}
                onDispatch={(id) => onUpdateEvidence(id, {
                  status: "Police Dispatched",
                  dispatchTime: new Date().toLocaleTimeString()
                })}
              />
            ))
          )}
        </div>
      </Block>

      {/* Device Intelligence */}
      <Block label="DEVICE INTELLIGENCE">
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {[
            ["CONNECTION SECURITY", "✓ SSL/TLS ENCRYPTED", "var(--safe)"],
            ["INFERENCE ENGINE", "✓ HYBRID (FE+BE)", "var(--safe)"],
            ["DETECTION TARGETS", "MALE · FEMALE · VIOLENCE · WEAPON", "var(--accent)"],
          ].map(([label, value, color]) => (
            <div key={label} style={{ background: "var(--bg3)", padding: "8px 10px", border: "1px solid var(--border)" }}>
              <div style={{ fontFamily: "monospace", fontSize: 7, color: "var(--text-dim)", marginBottom: 3 }}>{label}</div>
              <div style={{ fontFamily: "monospace", fontSize: 9, color }}>{value}</div>
            </div>
          ))}
        </div>
      </Block>

      {/* Footer */}
      <div style={{ marginTop: "auto", padding: 12, opacity: 0.4 }}>
        <div style={{ ...mono, fontSize: 7, color: "var(--text-dim)", lineHeight: 1.6 }}>
          SURAKSHADRISHTI CORE V3.0 · HYBRID AI<br />
          FRONTEND GENDER + BACKEND WEAPON/VIOLENCE
        </div>
      </div>
    </div>
  );
}

function Block({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ borderBottom: "1px solid var(--border)", padding: "14px 16px" }}>
      <div style={{ fontFamily: "monospace", fontSize: 8, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 12 }}>{label}</div>
      {children}
    </div>
  );
}
