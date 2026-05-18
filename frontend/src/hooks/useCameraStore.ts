"use client";
import { useEffect, useState } from "react";

export interface CameraEntry {
  id: string;
  type: "cctv" | "webcam";
  label: string;
  url?: string;       // CCTV stream URL (MJPEG / HLS)
  lat?: number;
  lng?: number;
  addedAt: string;
  status: "active" | "offline" | "error";
}

const KEY = "sd_cameras_v2";

function load(): CameraEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function save(cams: CameraEntry[]) {
  try { localStorage.setItem(KEY, JSON.stringify(cams)); } catch {}
}

export function useCameraStore() {
  const [cameras, setCameras] = useState<CameraEntry[]>([]);

  useEffect(() => { setCameras(load()); }, []);

  function addCamera(cam: Omit<CameraEntry, "id" | "addedAt">) {
    const entry: CameraEntry = {
      ...cam,
      id: `CAM-${Date.now()}`,
      addedAt: new Date().toISOString(),
    };
    setCameras(prev => {
      const next = [...prev, entry];
      save(next);
      return next;
    });
    return entry;
  }

  function removeCamera(id: string) {
    setCameras(prev => {
      const next = prev.filter(c => c.id !== id);
      save(next);
      return next;
    });
  }

  function updateStatus(id: string, status: CameraEntry["status"]) {
    setCameras(prev => {
      const next = prev.map(c => c.id === id ? { ...c, status } : c);
      save(next);
      return next;
    });
  }

  function updateCamera(id: string, updates: Partial<Omit<CameraEntry, "id" | "addedAt">>) {
    setCameras(prev => {
      const next = prev.map(c => c.id === id ? { ...c, ...updates } : c);
      save(next);
      return next;
    });
  }

  return { cameras, addCamera, removeCamera, updateStatus, updateCamera };
}
