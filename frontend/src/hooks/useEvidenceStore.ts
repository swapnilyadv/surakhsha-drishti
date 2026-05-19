"use client";
import { useEffect, useState } from "react";

export interface EvidenceEntry {
  id: string;
  cameraId: string;
  cameraLabel: string;
  timestamp: string;       // human-readable
  isoTime: string;         // ISO for sorting
  confidence: number;
  type: string;            // VIOLENCE, HARASSMENT, etc.
  thumbnail?: string;      // base64 data-URL
  videoUrl?: string;       // blob URL for session playback
  maleCount?: number;
  femaleCount?: number;
  weaponDetected?: boolean;
  weaponType?: string;     // Knife, Gun, etc.
  lat?: number;
  lng?: number;
  locationName?: string;
  status: "Active" | "Police Dispatched" | "Resolved" | "More Help Requested";
  authorityStation?: string;
  dispatchTime?: string;
}

const KEY = "sd_evidence_v2";

function load(): EvidenceEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function save(items: EvidenceEntry[]) {
  try {
    const slim = items.map(({ thumbnail: _t, ...rest }) => rest);
    localStorage.setItem(KEY, JSON.stringify(slim));
  } catch {}
}

export function useEvidenceStore() {
  const [evidence, setEvidence] = useState<EvidenceEntry[]>([]);

  useEffect(() => { 
    // 1. Fetch initial persistent records from backend database
    async function fetchFromBackend() {
      try {
        const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8765";
        const res = await fetch(`${backendUrl}/api/evidence`);
        if (res.ok) {
          const data: EvidenceEntry[] = await res.json();
          // map relative URLs to absolute backend domain path
          const mapped = data.map(item => ({
            ...item,
            videoUrl: item.videoUrl && item.videoUrl.startsWith("/") ? `${backendUrl}${item.videoUrl}` : item.videoUrl
          }));
          
          const wasSeeded = localStorage.getItem("sd_seeded");
          if (mapped.length > 0 || wasSeeded === "true") {
            setEvidence(mapped);
            save(mapped);
            return;
          }
        }
      } catch (err) {
        console.warn("Could not load evidence from backend database, using local cache fallback:", err);
      }
      
      const loaded = load();
      const wasSeeded = localStorage.getItem("sd_seeded");
      if (loaded.length === 0 && wasSeeded !== "true") {
        seedSampleEvidence();
        localStorage.setItem("sd_seeded", "true");
      } else {
        setEvidence(loaded); 
      }
    }
    
    fetchFromBackend();

    // 2. Handle Real-Time WS State Synchronization
    const handleWSDispatch = (e: CustomEvent) => {
      const { evidence_id, station, officer } = e.detail;
      setEvidence(prev => {
        const next = prev.map(item => item.id === evidence_id ? {
          ...item,
          status: "Police Dispatched" as const,
          authorityStation: station,
          dispatchTime: new Date().toLocaleTimeString()
        } : item);
        save(next);
        return next;
      });
    };

    const handleWSHelp = (e: CustomEvent) => {
      const { evidence_id, station, location } = e.detail;
      setEvidence(prev => {
        const next = prev.map(item => item.id === evidence_id ? {
          ...item,
          status: "More Help Requested" as const,
          lat: location.lat,
          lng: location.lng,
          authorityStation: station
        } : item);
        save(next);
        return next;
      });
    };

    const handleWSResolve = (e: CustomEvent) => {
      const { evidence_id } = e.detail;
      setEvidence(prev => {
        const next = prev.map(item => item.id === evidence_id ? {
          ...item,
          status: "Resolved" as const
        } : item);
        save(next);
        return next;
      });
    };

    window.addEventListener("ws-dispatch-accepted", handleWSDispatch as any);
    window.addEventListener("ws-need-more-help", handleWSHelp as any);
    window.addEventListener("ws-evidence-resolved", handleWSResolve as any);
    
    return () => {
      window.removeEventListener("ws-dispatch-accepted", handleWSDispatch as any);
      window.removeEventListener("ws-need-more-help", handleWSHelp as any);
      window.removeEventListener("ws-evidence-resolved", handleWSResolve as any);
    };
  }, []);

  function seedSampleEvidence() {
    const host = typeof window !== "undefined" ? 
      (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? '127.0.0.1' : window.location.hostname) 
      : '127.0.0.1';
    
    const samples: EvidenceEntry[] = [
      {
        id: "EVD-LIVE-00",
        cameraId: "CAM-01",
        cameraLabel: "LIVE WEBCAM TEST",
        timestamp: "REAL-TIME",
        isoTime: new Date().toISOString(),
        confidence: 0.99,
        type: "LIVE TEST",
        status: "Active",
        lat: 19.0760,
        lng: 72.8777,
        videoUrl: `http://${host}:8765/api/stream/mjpeg`,
        maleCount: 0,
        femaleCount: 0,
        weaponDetected: false,
      },
      {
        id: "EVD-PRO-01",
        cameraId: "CAM-01",
        cameraLabel: "Main Entrance CCTV",
        timestamp: new Date().toLocaleTimeString(),
        isoTime: new Date().toISOString(),
        confidence: 0.92,
        type: "HARASSMENT",
        status: "Active",
        lat: 19.0760,
        lng: 72.8777,
        videoUrl: `http://${host}:8765/api/stream/mjpeg`,
        maleCount: 2,
        femaleCount: 1,
        weaponDetected: false,
      },
      {
        id: "EVD-PRO-02",
        cameraId: "CAM-02",
        cameraLabel: "Parking Zone B",
        timestamp: new Date(Date.now() - 3600000).toLocaleTimeString(),
        isoTime: new Date(Date.now() - 3600000).toISOString(),
        confidence: 0.88,
        type: "WEAPON",
        status: "Active",
        lat: 19.0500,
        lng: 72.8300,
        videoUrl: `http://${host}:8765/api/stream/mjpeg`,
        weaponDetected: true,
        weaponType: "Knife",
      }
    ];
    setEvidence(samples);
    save(samples);
  }

  function addEvidence(entry: Partial<EvidenceEntry>) {
    const item: EvidenceEntry = {
      id: entry.id || `EVD-${Date.now()}`,
      cameraId: entry.cameraId || "unknown",
      cameraLabel: entry.cameraLabel || "Unknown Cam",
      timestamp: entry.timestamp || new Date().toLocaleTimeString(),
      isoTime: entry.isoTime || new Date().toISOString(),
      confidence: entry.confidence || 0,
      type: entry.type || "UNKNOWN",
      status: entry.status || "Active",
      lat: entry.lat || 19.0760, // Default to Mumbai Center if no GPS
      lng: entry.lng || 72.8777,
      locationName: entry.locationName || "Gateway of India Plaza",
      videoUrl: entry.videoUrl || undefined, 
      ...entry,
    } as EvidenceEntry;

    setEvidence(prev => {
      const next = [item, ...prev.filter(x => x.id !== item.id)];
      save(next);
      return next;
    });
    return item;
  }

  async function clearAll() {
    setEvidence([]);
    localStorage.removeItem(KEY);
    localStorage.setItem("sd_seeded", "true");
    
    // Clear backend as well
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8765";
      await fetch(`${backendUrl}/api/evidence/clear`, { method: "POST" });
    } catch {}
  }

  async function updateEvidence(id: string, updates: Partial<EvidenceEntry>) {
    // 1. Optimistic UI update locally
    setEvidence(prev => {
      const next = prev.map(item => item.id === id ? { ...item, ...updates } : item);
      save(next);
      return next;
    });

    // 2. Synchronize to Backend HTTP APIs
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8765";
      
      if (updates.status === "Resolved") {
        await fetch(`${backendUrl}/api/evidence/${id}/resolve`, { method: "POST" });
      } else if (updates.status === "Police Dispatched") {
        let stationName = updates.authorityStation || "Mumbai Police Station";
        try {
          const authStr = localStorage.getItem("sd_auth");
          if (authStr) {
            stationName = JSON.parse(authStr).user || stationName;
          }
        } catch {}

        await fetch(`${backendUrl}/api/evidence/${id}/dispatch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            station: stationName,
            officer: "Officer-" + Math.floor(Math.random() * 89 + 10)
          })
        });
      } else if (updates.status === "More Help Requested") {
        let stationName = updates.authorityStation || "Mumbai Police Station";
        try {
          const authStr = localStorage.getItem("sd_auth");
          if (authStr) {
            stationName = JSON.parse(authStr).user || stationName;
          }
        } catch {}

        await fetch(`${backendUrl}/api/evidence/${id}/escalate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            station: stationName,
            lat: updates.lat || 19.0760,
            lng: updates.lng || 72.8777
          })
        });
      }
    } catch (err) {
      console.warn("Failed to synchronize state update with backend API:", err);
    }
  }

  return { evidence, addEvidence, clearAll, updateEvidence, seedSampleEvidence };
}
