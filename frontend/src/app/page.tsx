"use client";
import { useCallback, useEffect, useState, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";

import LoginScreen      from "@/components/LoginScreen";
import TopBar           from "@/components/TopBar";
import CameraGrid       from "@/components/CameraGrid";
import StatsPanel       from "@/components/StatsPanel";
import EvidenceVault    from "@/components/EvidenceVault";
import LiveMap          from "@/components/LiveMap";
import AdminPanel       from "@/components/AdminPanel";
import Analysis         from "@/components/Analysis";
import CameraManagement from "@/components/CameraManagement";
import Toast, { useToast } from "@/components/Toast";

import { useCameraStore }   from "@/hooks/useCameraStore";
import { useEvidenceStore } from "@/hooks/useEvidenceStore";

type Tab = "dashboard" | "evidence" | "map" | "admin";

export default function Home() {
  const [loggedIn,    setLoggedIn]    = useState(false);
  const [currentUser, setCurrentUser] = useState("");
  const [isAdmin,     setIsAdmin]     = useState(false);
  const [tab,         setTab]         = useState<Tab>("dashboard");
  const [alertMsg,    setAlertMsg]    = useState("");

  const { toast, showToast, setToast } = useToast();
  const { cameras, addCamera, removeCamera, updateCamera } = useCameraStore();
  const { evidence, addEvidence, clearAll, updateEvidence } = useEvidenceStore();
  const [alertCamIds, setAlertCamIds] = useState<Set<string>>(new Set());
  const alarmRef = useRef<HTMLAudioElement | null>(null);
  const [liveMale, setLiveMale] = useState(0);
  const [liveFemale, setLiveFemale] = useState(0);

  const handleGenderUpdate = useCallback((male: number, female: number) => {
    setLiveMale(male);
    setLiveFemale(female);
  }, []);

  // Initialize audio and auth on mount
  useEffect(() => {
    alarmRef.current = new Audio("/sound/alarm.wav");
    alarmRef.current.loop = true;

    const saved = localStorage.getItem("sd_auth");
    if (saved) {
      const { user, isAdmin: savedIsAdmin, expiry } = JSON.parse(saved);
      if (Date.now() < expiry) {
        setCurrentUser(user);
        setIsAdmin(!!savedIsAdmin);
        setLoggedIn(true);
      } else {
        localStorage.removeItem("sd_auth");
      }
    }
  }, []);

  const handleLogin = (user: string, adminStatus: boolean) => {
    setCurrentUser(user);
    setIsAdmin(adminStatus);
    setLoggedIn(true);
    const expiry = Date.now() + 24 * 60 * 60 * 1000;
    localStorage.setItem("sd_auth", JSON.stringify({ user, isAdmin: adminStatus, expiry }));
  };

  const handleLogout = () => {
    setLoggedIn(false);
    setIsAdmin(false);
    stopAlarm();
    localStorage.removeItem("sd_auth");
  };

  const stopAlarm = () => {
    if (alarmRef.current) {
      alarmRef.current.pause();
      alarmRef.current.currentTime = 0;
    }
    setAlertMsg("");
  };

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4500);
    return () => clearTimeout(t);
  }, [toast, setToast]);

  const handleDetection = useCallback((entry: Parameters<typeof addEvidence>[0] & { cameraId: string }) => {
    const triggerAlarm = () => {
      setAlertCamIds(prev => new Set(prev).add(entry.cameraId));
      setAlertMsg(`VIOLENCE DETECTED — ${entry.cameraLabel} · ${((entry.confidence ?? 0) * 100).toFixed(0)}% confidence`);
      if (alarmRef.current) {
        alarmRef.current.play().catch(e => console.error("Audio play failed:", e));
      }
      setTimeout(() => {
        setAlertCamIds(prev => { const s = new Set(prev); s.delete(entry.cameraId); return s; });
      }, 15000);
    };

    if (typeof navigator !== "undefined" && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          addEvidence({ ...entry, lat: pos.coords.latitude, lng: pos.coords.longitude, locationName: "User Device Location" });
          triggerAlarm();
        },
        () => {
          addEvidence(entry);
          triggerAlarm();
        }
      );
    } else {
      addEvidence(entry);
      triggerAlarm();
    }
  }, [addEvidence]);

  useEffect(() => {
    const handler = (e: any) => {
      const { id, videoUrl } = e.detail;
      updateEvidence(id, { videoUrl });
    };
    window.addEventListener('update-evidence-video', handler);
    return () => window.removeEventListener('update-evidence-video', handler);
  }, [updateEvidence]);

  const handleAlert = useCallback((msg: string) => {
    showToast(msg, true);
  }, [showToast]);

  const isAlert = alertMsg.length > 0;

  return (
    <>
      <AnimatePresence>
        {!loggedIn && (
          <motion.div key="login" exit={{ opacity: 0 }} transition={{ duration: 0.4 }} style={{ position: "fixed", inset: 0, zIndex: 100 }}>
            <LoginScreen onLogin={handleLogin}/>
          </motion.div>
        )}
      </AnimatePresence>

      {loggedIn && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}
          style={{ display: "flex", flexDirection: "column", height: "100vh" }}>

          <TopBar 
            currentTab={tab} 
            onTabChange={t => setTab(t as Tab)} 
            isAlert={isAlert}
            onLogout={handleLogout} 
            currentUser={currentUser}
            isAdmin={isAdmin}
          />

          <AnimatePresence>
            {isAlert && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                style={{ background: "rgba(255,34,68,0.13)", borderBottom: "2px solid var(--danger)", padding: "9px 20px", fontFamily: "monospace", fontSize: 12, color: "var(--danger)", letterSpacing: 2, textTransform: "uppercase", display: "flex", alignItems: "center", gap: 14, flexShrink: 0, animation: "alert-flash-bg 0.5s infinite" }}>
                <span>⚠</span>
                <span>{alertMsg}</span>
                <button onClick={stopAlarm}
                  style={{ marginLeft: "auto", fontFamily: "monospace", fontSize: 10, background: "transparent", border: "1px solid var(--danger)", color: "var(--danger)", padding: "3px 12px", cursor: "pointer", letterSpacing: 1 }}>
                  ACKNOWLEDGE
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          <div style={{ flex: 1, overflow: "hidden" }}>
            {tab === "dashboard" && (
              <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
                <CameraGrid
                  cameras={cameras}
                  alertCamIds={alertCamIds}
                  onAddCamera={addCamera}
                  onRemoveCamera={removeCamera}
                  onDetection={handleDetection}
                  onAlert={handleAlert}
                  onGenderUpdate={handleGenderUpdate}
                />
                <StatsPanel
                  isAlert={isAlert}
                  evidence={evidence}
                  onUpdateEvidence={updateEvidence}
                  cameraCount={cameras.length}
                  activeCameraCount={cameras.filter(c => c.status === "active").length}
                  cameras={cameras}
                  alertCamIds={alertCamIds}
                  liveMale={liveMale}
                  liveFemale={liveFemale}
                />
              </div>
            )}

            {tab === "evidence" && (
              <EvidenceVault evidence={evidence} onClearAll={clearAll} onUpdate={updateEvidence}/>
            )}

            {tab === "map" && (
              <div style={{ height: "100%" }}>
                <LiveMap cameras={cameras} alertCamIds={alertCamIds} showToast={showToast} onUpdateCamera={updateCamera}/>
              </div>
            )}

            {tab === "admin" && (
              <AdminPanel cameras={cameras} onRemoveCamera={removeCamera} currentUser={currentUser}/>
            )}
          </div>
        </motion.div>
      )}

      <Toast toast={toast}/>
    </>
  );
}
