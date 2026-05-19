"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { CameraEntry } from "@/hooks/useCameraStore";
import { AnimatePresence, motion } from "framer-motion";
import CameraFeed from "./CameraFeed";

// Must be dynamically imported — Leaflet requires browser APIs (no SSR)
const MapClient = dynamic(() => import("./MapClient"), {
  ssr: false,
  loading: () => (
    <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: "#050d18", flexDirection: "column", gap: 12 }}>
      <div style={{ fontFamily: "monospace", fontSize: 12, color: "var(--text-dim)", letterSpacing: 2 }}>INITIALIZING TACTICAL MAP...</div>
    </div>
  ),
});

import { usePoliceStore } from "@/hooks/usePoliceStore";

interface Props {
  cameras: CameraEntry[];
  alertCamIds: Set<string>;
  showToast: (msg: string, danger?: boolean) => void;
  onUpdateCamera: (id: string, updates: Partial<CameraEntry>) => void;
}

export default function LiveMap({ cameras, alertCamIds, showToast, onUpdateCamera }: Props) {
  const { accounts: stations } = usePoliceStore();
  const [bigScreenCamId, setBigScreenCamId] = useState<string | null>(null);
  const bigScreenCam = cameras.find(c => c.id === bigScreenCamId);

  // Dynamic GPS for Webcam was removed as webcam type is no longer supported.
  useEffect(() => {
    // CCTV cameras have fixed locations.
  }, [cameras]);

  const hasCamsWithGPS = cameras.some(c => c.lat && c.lng);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", position: "relative" }}>
      {/* Legend / Control Bar */}
      <div style={{ padding: "8px 16px", borderBottom: "1px solid var(--border)", background: "var(--bg2)", display: "flex", alignItems: "center", gap: 16, flexShrink: 0 }}>
        <span style={{ fontFamily: "monospace", fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: "var(--text-dim)" }}>LIVE TACTICAL MAP</span>
        <div style={{ display: "flex", gap: 12, marginLeft: "auto", alignItems: "center" }}>
          {[
            { color: "#ffaa00", label: "CCTV (Fixed)" },
            { color: "#00ffaa", label: "Police Station" },
            { color: "#ff2244", label: "Alert" },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, fontFamily: "monospace", fontSize: 10, color: "var(--text-dim)" }}>
              <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, boxShadow: `0 0 6px ${color}` }}/>
              {label}
            </div>
          ))}
          <span style={{ fontFamily: "monospace", fontSize: 10, color: "var(--text-dim)", marginLeft: 10 }}>· {cameras.filter(c => c.lat && c.lng).length + stations.length} NODES MAPPED</span>
        </div>
      </div>

      {/* Map Content */}
      <div style={{ flex: 1, position: "relative" }}>
        <MapClient
          cameras={cameras}
          stations={stations}
          alertCamIds={alertCamIds}
          showToast={showToast}
          onMarkerClick={id => setBigScreenCamId(id)}
        />

        {/* GPS Warning Overlay */}
        {cameras.length > 0 && !hasCamsWithGPS && (
          <div style={{ position: "absolute", bottom: 20, left: "50%", transform: "translateX(-50%)", background: "rgba(8,13,26,0.92)", border: "1px solid var(--border)", fontFamily: "monospace", fontSize: 11, color: "var(--warning)", padding: "8px 16px", letterSpacing: 1, zIndex: 1000 }}>
            ⚠ NO GPS DATA — CONNECT CAMERAS WITH LOCATION ENABLED
          </div>
        )}
      </div>

      {/* Big Screen View Modal */}
      <AnimatePresence>
        {bigScreenCam && (
          <div style={{ position: "absolute", inset: 0, zIndex: 2000, background: "rgba(0,0,0,0.85)", display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}
            onClick={e => e.target === e.currentTarget && setBigScreenCamId(null)}>
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              style={{ width: "100%", maxWidth: 1000, background: "var(--bg2)", border: "1px solid var(--accent2)", boxShadow: "0 0 30px rgba(0,170,255,0.2)", position: "relative" }}
            >
              <div style={{ padding: "12px 20px", background: "var(--bg3)", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontFamily: "Orbitron,sans-serif", fontSize: 14, fontWeight: 700, color: "var(--accent)", letterSpacing: 2 }}>
                  LIVE FEED :: {bigScreenCam.label}
                </div>
                <button onClick={() => setBigScreenCamId(null)} style={{ background: "transparent", border: "none", color: "var(--text-dim)", fontSize: 24, cursor: "pointer", lineHeight: 1 }}>×</button>
              </div>
              <div style={{ padding: 10, width: "100%", aspectRatio: "16/9", overflow: "hidden", position: "relative" }}>
                 <CameraFeed
                    camera={bigScreenCam}
                    onRemove={() => {}} // Disabled in big view
                    onDetection={() => {}}
                    onAlert={msg => showToast(msg, true)}
                 />
              </div>
              <div style={{ padding: "10px 20px", display: "flex", justifyContent: "space-between", background: "var(--bg3)", borderTop: "1px solid var(--border)" }}>
                 <div style={{ fontFamily: "monospace", fontSize: 10, color: "var(--text-dim)" }}>
                    LAT: {bigScreenCam.lat?.toFixed(6)} · LNG: {bigScreenCam.lng?.toFixed(6)}
                 </div>
                 <div style={{ fontFamily: "monospace", fontSize: 10, color: "var(--safe)" }}>ENCRYPTED TACTICAL STREAM :: ACTIVE</div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
