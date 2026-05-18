"use client";
import dynamic from "next/dynamic";
import { CameraEntry } from "@/hooks/useCameraStore";

import { usePoliceStore } from "@/hooks/usePoliceStore";

const MapClient = dynamic(() => import("./MapClient"), {
  ssr: false,
  loading: () => <div style={{ background: "#050d18", height: "100%" }}/>
});

interface Props {
  cameras: CameraEntry[];
  alertCamIds: Set<string>;
}

export default function MiniMap({ cameras, alertCamIds }: Props) {
  const { accounts: stations } = usePoliceStore();

  return (
    <div style={{
      width: "100%", height: "100%", background: "var(--bg2)",
      position: "relative", overflow: "hidden"
    }}>
      <MapClient 
        cameras={cameras} 
        stations={stations}
        alertCamIds={alertCamIds} 
        showToast={() => {}} 
      />
      {/* Overlay to prevent interactions but show visual */}
      <div style={{ position: "absolute", inset: 0, zIndex: 1000, pointerEvents: "none" }} />
    </div>
  );
}
