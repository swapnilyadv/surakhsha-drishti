"use client";
import { useEffect, useState, useCallback } from "react";
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

interface TimelineEvent {
  id: string;
  cameraId: string;
  cameraLabel: string;
  timestamp: string;
  type: string;
  severity: string;
  details: string;
}

export default function StatsPanel({
  isAlert, evidence, cameraCount, activeCameraCount,
  cameras, alertCamIds, onUpdateEvidence,
  liveMale = 0, liveFemale = 0,
}: Props) {
  const activeIncidents = evidence.slice(0, 5);
  const [sessionStart] = useState(Date.now());
  const [uptime, setUptime] = useState("00:00:00");
  const [memoryUsage, setMemoryUsage] = useState<string>("N/A");
  const [eventsTimeline, setEventsTimeline] = useState<TimelineEvent[]>([]);

  // Real-time backend detection data
  const { detection, connected } = useBackendAI();

  const threatLevel = detection.threat_level || "LOW";
  const threatScore = Math.round((detection.threat_score || 0) * 100);
  const motionIntensity = Math.round((detection.motion_intensity || 0) * 100);

  const violenceDetected = detection.violence_detected;
  const weaponDetected = detection.weapon_detected;
  const isAnyAlert = isAlert || violenceDetected || weaponDetected || threatLevel === "CRITICAL" || threatLevel === "HIGH";

  // Fetch real-time visual events timeline from backend
  const fetchEventsTimeline = useCallback(async () => {
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8765";
      const res = await fetch(`${backendUrl}/api/events/timeline`);
      if (res.ok) {
        const data = await res.json();
        setEventsTimeline(data.slice(0, 6)); // Display latest 6 events
      }
    } catch (e) {
      console.warn("Failed to fetch events timeline:", e);
    }
  }, []);

  useEffect(() => {
    fetchEventsTimeline();
    const interval = setInterval(fetchEventsTimeline, 2000);
    return () => clearInterval(interval);
  }, [fetchEventsTimeline]);

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

  // Resolve threat styling configurations
  const getThreatStyles = () => {
    switch (threatLevel) {
      case "CRITICAL":
        return { color: "#ef4444", bg: "rgba(239, 68, 68, 0.15)", text: "CRITICAL THREAT" };
      case "HIGH":
        return { color: "#f97316", bg: "rgba(249, 115, 22, 0.12)", text: "HIGH RISK" };
      case "MEDIUM":
        return { color: "#eab308", bg: "rgba(234, 179, 8, 0.10)", text: "SUSPICIOUS" };
      default:
        return { color: "#10b981", bg: "rgba(16, 185, 129, 0.08)", text: "NOMINAL" };
    }
  };

  const threatStyles = getThreatStyles();

  return (
    <div style={{ width: 300, flexShrink: 0, display: "flex", flexDirection: "column", overflowY: "auto", background: "var(--bg2)", borderLeft: "1px solid var(--border)" }}>

      {/* Mini Map */}
      <div style={{ borderBottom: "1px solid var(--border)", background: "#050810" }}>
        <div style={{ padding: "8px 14px", ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 1 }}>TACTICAL OVERVIEW</div>
        <div style={{ width: "100%", height: 140 }}>
          <MiniMap cameras={cameras} alertCamIds={alertCamIds} />
        </div>
      </div>

      {/* ── DYNAMIC THREAT SCORE ENGINE (Phase 2 & 6) ── */}
      <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)", background: threatStyles.bg }}>
        <div style={{ ...mono, fontSize: 8, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: threatStyles.color,
            flexShrink: 0,
            animation: threatLevel !== "LOW" ? "pulse-dot 0.8s infinite alternate" : "none"
          }} />
          ACTIVE THREAT STATE
        </div>
        
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 8 }}>
          <div style={{
            fontFamily: "Orbitron,sans-serif",
            fontSize: 26,
            fontWeight: 900,
            lineHeight: 1,
            color: threatStyles.color,
            textShadow: `0 0 16px ${threatStyles.color}77`
          }}>
            {threatLevel}
          </div>
          <div style={{ ...mono, fontSize: 10, color: "var(--text)", fontWeight: "bold" }}>
            {threatScore}%
          </div>
        </div>

        {/* Progress Bar */}
        <div style={{ height: 6, background: "rgba(0,0,0,0.4)", borderRadius: 3, overflow: "hidden", border: "1px solid rgba(255,255,255,0.04)" }}>
          <div style={{
            height: "100%",
            width: `${Math.max(threatScore, 2)}%`,
            background: threatStyles.color,
            boxShadow: `0 0 10px ${threatStyles.color}`,
            transition: "width 0.5s cubic-bezier(0.4,0,0.2,1)"
          }} />
        </div>

        {/* Signal telemetries */}
        <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between", ...mono, fontSize: 8, color: "var(--text-dim)" }}>
          <span>MOTION INTENSITY</span>
          <span style={{ color: motionIntensity > 50 ? "var(--warning)" : "var(--text-dim)" }}>{motionIntensity}%</span>
        </div>
      </div>

      {/* ── SESSION STATUS (Upgraded Telemetries) ── */}
      <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ ...mono, fontSize: 8, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 8 }}>SESSION STATUS</div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 26, fontWeight: 900, lineHeight: 1, color: isAnyAlert ? "var(--danger)" : "var(--safe)", textShadow: isAnyAlert ? "0 0 18px rgba(255,34,68,0.5)" : "0 0 18px rgba(0,255,136,0.3)" }}>
            {isAnyAlert ? "ALERT" : "STABLE"}
          </div>
          <span style={{ ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 1 }}>
            {detection.action ? detection.action.toUpperCase() : (violenceDetected ? "VIOLENCE ACTIVE" : weaponDetected ? "WEAPON DETECTED" : "SECURE")}
          </span>
        </div>

        <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          {[
            ["SESSION UPTIME", uptime],
            ["CAMERAS ACTIVE", `${activeCameraCount} / ${cameraCount}`],
            ["PIPELINE TELEMETRY", `${detection.fps || 20} FPS`],
            ["STREAM STATUS", connected ? "✓ CONNECTED" : "✗ RECONNECTING"],
          ].map(([l, v]) => (
            <div key={l} style={{ background: "rgba(255,255,255,0.02)", padding: "7px 10px", border: "1px solid var(--border)", borderRadius: 2 }}>
              <div style={{ ...mono, fontSize: 7, color: "var(--text-dim)", letterSpacing: 1, marginBottom: 4 }}>{l}</div>
              <div style={{ ...mono, fontSize: 10, color: l === "STREAM STATUS" && !connected ? "var(--danger)" : "var(--text)", fontWeight: 600 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── LIVE GENDER COUNT ── */}
      <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ ...mono, fontSize: 8, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 4, height: 4, borderRadius: "50%", background: totalPeople > 0 ? "#00aaff" : "var(--text-dim)", animation: totalPeople > 0 ? "pulse-dot 1.5s infinite" : "none", flexShrink: 0 }} />
          LIVE POPULATION SCAN
        </div>

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
      </div>

      {/* ── REAL-TIME TEMPORAL EVENTS TIMELINE (Phase 5 & 6) ── */}
      <Block label="SURVEILLANCE ACTIVITY LOG">
        <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 190, overflowY: "auto", paddingRight: 4 }}>
          {eventsTimeline.length === 0 ? (
            <div style={{ padding: "14px 0", textAlign: "center", border: "1px dashed rgba(255,255,255,0.05)", borderRadius: 2 }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 1 }}>NO ANOMALOUS INCIDENTS</div>
            </div>
          ) : (
            eventsTimeline.map((evt, idx) => {
              const borderCol = evt.severity === "CRITICAL" ? "rgba(239, 68, 68, 0.4)" : "rgba(255, 255, 255, 0.08)";
              const textCol = evt.severity === "CRITICAL" ? "#ef4444" : "var(--text)";
              const uniqueKey = `${evt.id}-${evt.timestamp}-${idx}`;
              return (
                <div key={uniqueKey} style={{ background: "rgba(255,255,255,0.01)", border: `1px solid ${borderCol}`, padding: "8px 10px", borderRadius: 3 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", ...mono, fontSize: 8, marginBottom: 4 }}>
                    <span style={{ color: textCol, fontWeight: "bold" }}>{evt.type}</span>
                    <span style={{ color: "var(--text-dim)" }}>{evt.timestamp}</span>
                  </div>
                  <div style={{ fontSize: 9, color: "var(--text-dim)", lineHeight: 1.4 }}>{evt.details}</div>
                  <div style={{ ...mono, fontSize: 7, color: "rgba(255,255,255,0.2)", marginTop: 4 }}>CAM: {evt.cameraLabel}</div>
                </div>
              );
            })
          )}
        </div>
      </Block>

      {/* Live Incident Tracker */}
      <Block label="LIVE EVIDENCE PIPELINE">
        <div className="space-y-4 max-h-[350px] overflow-y-auto pr-1">
          {activeIncidents.length === 0 ? (
            <div className="py-8 text-center border border-dashed border-zinc-800">
              <div className="text-zinc-600 text-[10px] font-mono tracking-widest">SYSTEM SECURE</div>
              <div className="text-zinc-800 text-[8px] font-mono mt-1 uppercase">No active recordings</div>
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
      <Block label="SECURITY SHIELD INTELLIGENCE">
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {[
            ["TRANSMISSION GATEWAY", "✓ SECURE REAL-TIME WEBSOCKET", "var(--safe)"],
            ["INTELLIGENCE AGENTS", "YOLOv8 + MEDIAPIPE + TEMPORAL CLASSIFIERS", "var(--accent)"],
            ["RECORDING BUFFER", "5s ROLLING PRE-BUFFER + 10s INCIDENT MERGE", "var(--accent)"],
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
          SURAKSHADRISHTI CORE V4.0 · PRODUCTION STACK<br />
          TEMPORAL VISION ENGINE + THREAT COORDINATION
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
