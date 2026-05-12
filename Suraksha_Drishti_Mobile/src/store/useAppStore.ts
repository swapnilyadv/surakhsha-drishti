import { create } from 'zustand';

export interface Detection {
  cam_id: string;
  violence: boolean;
  weapon: boolean;
  weapon_type?: string;
  confidence: number;
  timestamp: number;
  crowd_count: number;
  density: number;
  density_level: string;
}

export interface Incident {
  id: string;
  camId: string;
  location: string;
  ts: string;
  level: 'LOW' | 'HIGH' | 'CRITICAL';
  confidence: number;
  weapon?: string;
  summary: string;
  status: string;
}

interface AppState {
  isConnected: boolean;
  threatLevel: 'SAFE' | 'LOW' | 'HIGH' | 'CRITICAL';
  detections: Record<string, Detection>;
  incidents: Incident[];
  setConnected: (v: boolean) => void;
  setThreatLevel: (v: 'SAFE' | 'LOW' | 'HIGH' | 'CRITICAL') => void;
  updateDetection: (camId: string, d: Detection) => void;
  addIncident: (inc: Incident) => void;
  clearHistory: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  isConnected: false,
  threatLevel: 'SAFE',
  detections: {},
  incidents: [],
  setConnected: (v) => set({ isConnected: v }),
  setThreatLevel: (v) => set({ threatLevel: v }),
  updateDetection: (camId, d) => 
    set((state) => ({ 
      detections: { ...state.detections, [camId]: d } 
    })),
  addIncident: (inc) => 
    set((state) => ({ 
      incidents: [inc, ...state.incidents].slice(0, 50) 
    })),
  clearHistory: () => set({ detections: {}, incidents: [] }),
}));
