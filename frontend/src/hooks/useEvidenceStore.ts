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
  // strip thumbnails before saving to keep localStorage light
  try {
    const slim = items.map(({ thumbnail: _t, ...rest }) => rest);
    localStorage.setItem(KEY, JSON.stringify(slim));
  } catch {}
}

export function useEvidenceStore() {
  const [evidence, setEvidence] = useState<EvidenceEntry[]>([]);

  useEffect(() => { 
    const loaded = load();
    if (loaded.length === 0) {
      seedSampleEvidence();
    } else {
      setEvidence(loaded); 
    }
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
        locationName: "Local Tactical Unit (Webcam)",
        videoUrl: `http://${host}:8000/api/stream/mjpeg?camera_id=CAM-01&quality=50`,
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
        locationName: "Gateway of India Plaza",
        videoUrl: undefined,
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
        status: "Police Dispatched",
        lat: 19.0820,
        lng: 72.8890,
        locationName: "Bandstand Promenade",
        videoUrl: undefined,
        weaponDetected: true,
        weaponType: "Knife",
        authorityStation: "Bandra West Station",
        dispatchTime: new Date(Date.now() - 3000000).toLocaleTimeString(),
      }
    ];
    setEvidence(samples);
    save(samples);
  }

  function addEvidence(entry: Partial<EvidenceEntry>) {
    const item: EvidenceEntry = {
      id: `EVD-${Date.now()}`,
      cameraId: entry.cameraId || "unknown",
      cameraLabel: entry.cameraLabel || "Unknown Cam",
      timestamp: entry.timestamp || new Date().toLocaleTimeString(),
      isoTime: entry.isoTime || new Date().toISOString(),
      confidence: entry.confidence || 0,
      type: entry.type || "UNKNOWN",
      status: entry.status || "Active",
      lat: entry.lat || 19.0760, // Default to Mumbai Center if no GPS
      lng: entry.lng || 72.8777,
      locationName: entry.locationName || "Tactical Unit Location",
      videoUrl: entry.videoUrl || undefined, // Real recordings from MediaRecorder only
      ...entry,
    } as EvidenceEntry;

    setEvidence(prev => {
      const next = [item, ...prev];
      save(next);
      return next;
    });
    return item;
  }

  function clearAll() {
    setEvidence([]);
    localStorage.removeItem(KEY);
  }

  function updateEvidence(id: string, updates: Partial<EvidenceEntry>) {
    setEvidence(prev => {
      const next = prev.map(item => item.id === id ? { ...item, ...updates } : item);
      save(next);
      return next;
    });
  }

  return { evidence, addEvidence, clearAll, updateEvidence, seedSampleEvidence };
}
