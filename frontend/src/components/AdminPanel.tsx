"use client";
import { useState } from "react";
import { CameraEntry } from "@/hooks/useCameraStore";
import { usePoliceStore } from "@/hooks/usePoliceStore";

interface Props {
  cameras: CameraEntry[];
  onRemoveCamera: (id: string) => void;
  currentUser: string;
}

export default function AdminPanel({ cameras, onRemoveCamera, currentUser }: Props) {
  const { accounts, addAccount, removeAccount } = usePoliceStore();
  const [newName, setNewName] = useState("");
  const [newLoc, setNewLoc] = useState("");
  const [newPass, setNewPass] = useState("");

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  // Real browser / device info
  const browserInfo = typeof navigator !== "undefined" ? navigator.userAgent.split(" ").slice(-2).join(" ") : "Unknown";
  const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
  const camCount = cameras.length;
  const onlineCount = cameras.filter(c => c.status === "active").length;

  return (
    <div style={{ display: "flex", flexDirection: "column", padding: 20, gap: 16, overflowY: "auto", height: "100%" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 16, fontWeight: 700, letterSpacing: 4, color: "var(--accent)" }}>SYSTEM ADMIN</div>
        <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)" }}>SESSION: {currentUser}</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Police Station Services */}
        <AdminCard title="POLICE STATION SERVICES" style={{ gridColumn: "1 / -1" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 20 }}>
            <div style={{ borderRight: "1px solid var(--border)", paddingRight: 20 }}>
              <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 12 }}>Authorized stations currently on the network.</div>
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
               <div style={{ ...mono, fontSize: 10, color: "var(--accent2)", letterSpacing: 1 }}>CREATE NEW STATION ACCOUNT</div>
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

        {/* Processing */}
        <AdminCard title="PROCESSING">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>AI MODULES</div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text-bright)" }}>DISABLED</div>
            </div>
            <div style={{ background: "var(--bg3)", padding: "10px", border: "1px solid var(--border)" }}>
              <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4 }}>INFERENCE</div>
              <div style={{ ...mono, fontSize: 10, color: "var(--text-bright)" }}>NONE (NO AI)</div>
            </div>
          </div>
        </AdminCard>

        {/* Cameras */}
        <AdminCard title={`CAMERAS (${camCount})`}>
          {cameras.map(cam => (
            <div key={cam.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid rgba(26,40,64,0.3)" }}>
              <div style={{ fontSize: 11, color: "var(--text)" }}>{cam.label}</div>
              <button onClick={() => onRemoveCamera(cam.id)} style={{ ...mono, fontSize: 8, background: "transparent", border: "1px solid var(--danger)", color: "var(--danger)", padding: "2px 6px", cursor: "pointer" }}>DEL</button>
            </div>
          ))}
        </AdminCard>

        {/* System Info */}
        <AdminCard title="SYSTEM INFO" style={{ gridColumn: "1 / -1" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20 }}>
            {[
              ["App Version", "SurakshaDrishti v2.4.1"],
              ["Cameras Active", `${onlineCount} / ${camCount}`],
              ["Inference", "Disabled (No AI)"],
              ["Connection", isHttps ? "Secure (HTTPS)" : "Standard (HTTP)"],
              ["Storage", "Local Persistence"],
              ["Environment", browserInfo],
            ].map(([l, v]) => (
              <div key={l}>
                <div style={{ fontSize: 9, color: "var(--text-dim)", textTransform: "uppercase" }}>{l}</div>
                <div style={{ ...mono, fontSize: 10, color: "var(--text-bright)" }}>{v}</div>
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
