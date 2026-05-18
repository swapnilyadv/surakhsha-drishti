"use client";
import { useEffect, useState } from "react";

interface Props {
  currentTab: string;
  onTabChange: (tab: string) => void;
  isAlert: boolean;
  onLogout: () => void;
  currentUser: string;
  isAdmin?: boolean;
}

export default function TopBar({ currentTab, onTabChange, isAlert, onLogout, currentUser, isAdmin }: Props) {
  const [time, setTime] = useState("");

  const TABS = ["dashboard", "evidence", "map"];
  const TAB_LABELS = ["Dashboard", "Evidence", "Live Map"];

  if (isAdmin) {
    TABS.push("admin");
    TAB_LABELS.push("Admin");
  }

  useEffect(() => {
    const tick = () => setTime(new Date().toLocaleTimeString("en-GB"));
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{
      height: 52, background: "var(--bg2)", borderBottom: "1px solid var(--border)",
      display: "flex", alignItems: "center", padding: "0 20px", gap: 20, flexShrink: 0,
    }}>
      {/* Logo */}
      <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 15, fontWeight: 900, letterSpacing: 3, color: "var(--accent)", textShadow: "0 0 12px rgba(0,170,255,0.4)", whiteSpace: "nowrap" }}>
        ⬡ SURAKSHADRISHTI
      </div>

      <div style={{ width: 1, height: 24, background: "var(--border)" }} />

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4 }}>
        {TABS.map((tab, i) => (
          <button
            key={tab}
            onClick={() => onTabChange(tab)}
            style={{
              padding: "6px 16px", fontFamily: "monospace", fontSize: 11, letterSpacing: 1.5,
              textTransform: "uppercase", cursor: "pointer", border: "1px solid",
              background: currentTab === tab ? "rgba(0,100,200,0.1)" : "transparent",
              color: currentTab === tab ? "var(--accent)" : "var(--text-dim)",
              borderColor: currentTab === tab ? "var(--accent2)" : "transparent",
              transition: "all 0.15s",
            }}
          >
            {TAB_LABELS[i]}
          </button>
        ))}
      </div>

      {/* Right side */}
      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 16 }}>
        {/* Status pill */}
        <div style={{
          fontFamily: "monospace", fontSize: 10, letterSpacing: 1.5, padding: "4px 12px",
          border: `1px solid ${isAlert ? "var(--danger)" : "var(--safe)"}`,
          color: isAlert ? "var(--danger)" : "var(--safe)",
          background: isAlert ? "rgba(255,34,68,0.1)" : "rgba(0,255,136,0.06)",
          display: "flex", alignItems: "center", gap: 6,
          animation: isAlert ? "blink 0.8s infinite" : "none",
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: "50%", background: "currentColor",
            animation: "pulse-dot 2s infinite",
          }} />
          <span>{isAlert ? "ALERT" : "ALL CLEAR"}</span>
        </div>

        {/* Clock */}
        <div style={{ fontFamily: "monospace", fontSize: 13, color: "var(--text-dim)", letterSpacing: 1 }}>
          {time}
        </div>

        {/* User */}
        <div style={{ fontFamily: "monospace", fontSize: 10, color: "var(--text-dim)", letterSpacing: 1 }}>
          {currentUser}
        </div>

        {/* Logout */}
        <button
          onClick={onLogout}
          style={{
            fontFamily: "monospace", fontSize: 10, letterSpacing: 2, color: "var(--text-dim)",
            background: "transparent", border: "1px solid var(--border)", padding: "5px 12px",
            cursor: "pointer", textTransform: "uppercase", transition: "all 0.15s",
          }}
          onMouseEnter={e => { const b = e.target as HTMLElement; b.style.color = "var(--danger)"; b.style.borderColor = "var(--danger)"; }}
          onMouseLeave={e => { const b = e.target as HTMLElement; b.style.color = "var(--text-dim)"; b.style.borderColor = "var(--border)"; }}
        >
          LOGOUT
        </button>
      </div>
    </div>
  );
}
