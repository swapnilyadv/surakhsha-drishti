"use client";
import { motion, AnimatePresence } from "framer-motion";
import { EvidenceEntry } from "@/hooks/useEvidenceStore";

interface Props {
  incident: EvidenceEntry;
  onDispatch: (id: string) => void;
}

export default function LiveIncidentCard({ incident, onDispatch }: Props) {
  const isCritical = incident.type === "VIOLENCE" || incident.weaponDetected;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className={`relative mb-3 bg-zinc-900 border ${isCritical ? 'border-red-600/50 shadow-[0_0_15px_rgba(255,34,68,0.2)]' : 'border-zinc-800'}`}
    >
      {/* Visual Header */}
      <div className="relative aspect-video overflow-hidden">
        {incident.thumbnail && (
          <img src={incident.thumbnail} alt="Incident" className="w-full h-full object-cover" />
        )}
        <div className="absolute top-2 left-2 bg-black/80 px-2 py-0.5 text-[8px] font-mono text-white tracking-widest uppercase">
          {incident.cameraLabel}
        </div>
        <div className={`absolute top-2 right-2 px-2 py-0.5 text-[8px] font-mono text-white tracking-widest uppercase animate-pulse ${isCritical ? 'bg-red-600' : 'bg-orange-600'}`}>
          {isCritical ? "CRITICAL" : "MODERATE"} THREAT
        </div>
      </div>

      {/* Details Section */}
      <div className="p-3 space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-[10px] font-bold text-white uppercase">{incident.type}</span>
          <span className="text-[9px] font-mono text-zinc-500">{incident.timestamp}</span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[9px] font-mono text-zinc-400">
          <div className="bg-zinc-800/50 p-1.5 border border-zinc-700/50">
            <span className="block text-zinc-600 text-[7px] mb-0.5">POPULATION</span>
            <span className="text-blue-400">{incident.maleCount}M / {incident.femaleCount}F</span>
          </div>
          <div className="bg-zinc-800/50 p-1.5 border border-zinc-700/50">
            <span className="block text-zinc-600 text-[7px] mb-0.5">WEAPON</span>
            <span className={incident.weaponDetected ? "text-red-400" : "text-green-400"}>
              {incident.weaponDetected ? "DETECTED" : "NONE"}
            </span>
          </div>
        </div>

        <div className="bg-zinc-800/30 p-2 text-[9px] text-zinc-300 leading-relaxed italic border-l-2 border-zinc-600">
          "Potential conflict detected near {incident.cameraLabel}. AI confidence: {(incident.confidence * 100).toFixed(0)}%."
        </div>

        {/* Action Button */}
        <button
          onClick={() => onDispatch(incident.id)}
          disabled={incident.status === "Police Dispatched"}
          className={`w-full py-2 text-[10px] font-bold tracking-widest transition-all ${
            incident.status === "Police Dispatched" 
            ? "bg-blue-600/20 text-blue-400 border border-blue-600/30 cursor-default" 
            : "bg-red-600 hover:bg-red-500 text-white shadow-lg"
          }`}
        >
          {incident.status === "Police Dispatched" ? "POLICE DISPATCHED" : "DISPATCH EMERGENCY RESPONSE"}
        </button>
      </div>

      {/* Status Bar */}
      <div className={`h-0.5 w-full ${isCritical ? 'bg-red-600' : 'bg-orange-600'}`} />
    </motion.div>
  );
}
