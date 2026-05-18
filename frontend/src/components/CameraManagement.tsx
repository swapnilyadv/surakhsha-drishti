"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CameraEntry } from "@/hooks/useCameraStore";
import AddCameraModal from "./AddCameraModal";

interface Props {
  cameras: CameraEntry[];
  onAddCamera: (cam: Omit<CameraEntry, "id" | "addedAt">) => void;
  onRemoveCamera: (id: string) => void;
  onEditCamera: (id: string, updates: Partial<Omit<CameraEntry, "id" | "addedAt">>) => void;
}

export default function CameraManagement({ cameras, onAddCamera, onRemoveCamera, onEditCamera }: Props) {
  const [showAdd, setShowAdd] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editForm, setEditForm] = useState<Partial<CameraEntry>>({});

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  function startEdit(cam: CameraEntry) {
    setEditingId(cam.id);
    setEditForm({ label: cam.label, url: cam.url, lat: cam.lat, lng: cam.lng, status: cam.status });
  }

  function saveEdit(id: string) {
    onEditCamera(id, editForm);
    setEditingId(null);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Header */}
      <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border)", background: "var(--bg2)", display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
        <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 15, fontWeight: 700, letterSpacing: 4, color: "var(--accent)" }}>
          CAMERA MANAGEMENT
        </div>
        <div style={{ ...mono, fontSize: 10, color: "var(--text-dim)" }}>{cameras.length} REGISTERED</div>
        <button onClick={() => setShowAdd(true)}
          style={{ marginLeft: "auto", ...mono, fontSize: 10, letterSpacing: 1, background: "rgba(0,100,200,0.15)", border: "1px solid var(--accent2)", color: "var(--accent)", padding: "6px 16px", cursor: "pointer", textTransform: "uppercase" }}>
          + ADD CAMERA
        </button>
      </div>

      {/* Stats row */}
      <div style={{ padding: "12px 20px", borderBottom: "1px solid var(--border)", display: "flex", gap: 20, flexShrink: 0, background: "var(--bg3)" }}>
        {[
          ["TOTAL",    cameras.length,                          "var(--accent)"],
          ["CCTV",     cameras.filter(c => c.type === "cctv").length,   "var(--warning)"],
          ["WITH GPS", cameras.filter(c => c.lat && c.lng).length,      "var(--safe)"],
          ["ACTIVE",   cameras.filter(c => c.status === "active").length,"var(--safe)"],
        ].map(([l, v, col]) => (
          <div key={String(l)} style={{ textAlign: "center", minWidth: 60 }}>
            <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 24, fontWeight: 900, color: String(col) }}>{v}</div>
            <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", letterSpacing: 2, marginTop: 2 }}>{l}</div>
          </div>
        ))}
      </div>

      {/* Camera list */}
      <div style={{ flex: 1, overflowY: "auto", padding: 20 }}>
        {cameras.length === 0 ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60%", gap: 16, opacity: 0.5 }}>
            <div style={{ fontSize: 48 }}>📷</div>
            <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 14, color: "var(--text-dim)", letterSpacing: 4 }}>NO CAMERAS REGISTERED</div>
            <div style={{ ...mono, fontSize: 11, color: "var(--text-dim)" }}>Click + ADD CAMERA to register your first camera</div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {cameras.map(cam => (
              <motion.div key={cam.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                style={{ background: "var(--panel)", border: "1px solid var(--border)", borderLeft: `3px solid var(--warning)`, overflow: "hidden" }}>

                {editingId === cam.id ? (
                  /* Edit form */
                  <div style={{ padding: 16 }}>
                    <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 2, marginBottom: 12 }}>EDITING: {cam.id}</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
                      <div>
                        <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4, letterSpacing: 1 }}>LABEL</div>
                        <input value={editForm.label || ""} onChange={e => setEditForm(f => ({ ...f, label: e.target.value }))}
                          style={{ width: "100%", background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "7px 9px", outline: "none" }}/>
                      </div>
                      <div>
                        <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4, letterSpacing: 1 }}>STREAM URL</div>
                        <input value={editForm.url || ""} onChange={e => setEditForm(f => ({ ...f, url: e.target.value }))}
                          style={{ width: "100%", background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 11, padding: "7px 9px", outline: "none" }}/>
                      </div>
                      <div>
                        <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4, letterSpacing: 1 }}>LATITUDE</div>
                        <input value={editForm.lat ?? ""} onChange={e => setEditForm(f => ({ ...f, lat: parseFloat(e.target.value) || undefined }))}
                          placeholder="e.g. 19.0760"
                          style={{ width: "100%", background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "7px 9px", outline: "none" }}/>
                      </div>
                      <div>
                        <div style={{ ...mono, fontSize: 8, color: "var(--text-dim)", marginBottom: 4, letterSpacing: 1 }}>LONGITUDE</div>
                        <input value={editForm.lng ?? ""} onChange={e => setEditForm(f => ({ ...f, lng: parseFloat(e.target.value) || undefined }))}
                          placeholder="e.g. 72.8777"
                          style={{ width: "100%", background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 12, padding: "7px 9px", outline: "none" }}/>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button onClick={() => setEditingId(null)} style={{ ...mono, fontSize: 10, padding: "7px 14px", background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", cursor: "pointer" }}>CANCEL</button>
                      <button onClick={() => saveEdit(cam.id)} style={{ ...mono, fontSize: 10, padding: "7px 20px", background: "var(--accent2)", border: "none", color: "#fff", cursor: "pointer", fontWeight: 700 }}>SAVE CHANGES</button>
                    </div>
                  </div>
                ) : (
                  /* View mode */
                  <div style={{ padding: "14px 16px", display: "flex", alignItems: "center", gap: 14 }}>
                    <div style={{ fontSize: 24 }}>📡</div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-bright)" }}>{cam.label}</div>
                        <div style={{ ...mono, fontSize: 8, padding: "2px 7px", border: "1px solid", color: "var(--warning)", borderColor: "rgba(255,170,0,0.4)" }}>
                          CCTV
                        </div>
                        <div style={{ ...mono, fontSize: 8, padding: "2px 7px", border: "1px solid rgba(0,255,136,0.4)", color: "var(--safe)" }}>ACTIVE</div>
                      </div>
                      <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", display: "flex", gap: 14, flexWrap: "wrap" }}>
                        <span>ID: {cam.id}</span>
                        {cam.lat && cam.lng && <span>📍 {cam.lat.toFixed(4)}°N, {cam.lng.toFixed(4)}°E</span>}
                        {cam.url && <span style={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>🔗 {cam.url}</span>}
                        <span>Added: {new Date(cam.addedAt).toLocaleDateString("en-IN")}</span>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button onClick={() => startEdit(cam)}
                        style={{ ...mono, fontSize: 9, padding: "6px 12px", background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", cursor: "pointer", textTransform: "uppercase" }}>
                        ✎ EDIT
                      </button>
                      <button onClick={() => { if (confirm(`Remove "${cam.label}"?`)) onRemoveCamera(cam.id); }}
                        style={{ ...mono, fontSize: 9, padding: "6px 12px", background: "rgba(255,34,68,0.08)", border: "1px solid rgba(255,34,68,0.35)", color: "var(--danger)", cursor: "pointer", textTransform: "uppercase" }}>
                        🗑 DELETE
                      </button>
                    </div>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        )}
      </div>

      <AnimatePresence>
        {showAdd && (
          <AddCameraModal onAdd={cam => onAddCamera({ ...cam, status: "active" })} onClose={() => setShowAdd(false)}/>
        )}
      </AnimatePresence>
    </div>
  );
}
