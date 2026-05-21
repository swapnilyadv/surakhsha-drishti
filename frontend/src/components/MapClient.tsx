"use client";
// This file is dynamically imported with ssr:false — safe to use Leaflet here
import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { CameraEntry } from "@/hooks/useCameraStore";
import { PoliceStation } from "@/hooks/usePoliceStore";

// Fix leaflet default icon paths broken by webpack
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

// Custom icons
function makeIcon(color: string, blinking: boolean = false, shape: 'drop' | 'circle' = 'drop') {
  const anim = blinking ? 'animation: marker-pulse 0.8s infinite ease-in-out;' : '';
  const borderRadius = shape === 'circle' ? '50%' : '50% 50% 50% 0';
  const rotate = shape === 'circle' ? '0deg' : '-45deg';
  const glowShadow = blinking ? `box-shadow: 0 0 20px 6px ${color}cc;` : `box-shadow: 0 0 12px ${color}88;`;
  
  return L.divIcon({
    className: "",
    html: `
      <div style="position: relative; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;">
        ${blinking ? `
          <div style="
            position: absolute;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: rgba(255, 34, 68, 0.15);
            border: 1px dashed #ff2244;
            animation: spin 4s linear infinite;
            top: -7px;
            left: -7px;
            pointer-events: none;
            box-shadow: inset 0 0 8px rgba(255, 34, 68, 0.2);
          "></div>
          <div style="
            position: absolute;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: transparent;
            border: 2px solid rgba(255, 34, 68, 0.4);
            animation: sonar-pulse 1.2s infinite ease-out;
            top: -13px;
            left: -13px;
            pointer-events: none;
          "></div>
        ` : ''}
        <div style="
          width:20px;height:20px;border-radius:${borderRadius};
          background:${color};border:2px solid rgba(255,255,255,0.9);
          transform:rotate(${rotate}); ${glowShadow}
          ${anim}
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        "></div>
      </div>
    `,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -15],
  });
}

const iconStation = makeIcon("#00ffaa", false, 'circle');

function FitBounds({ cameras, stations }: { cameras: CameraEntry[], stations: PoliceStation[] }) {
  const map = useMap();
  useEffect(() => {
    const validC = cameras.filter(c => c.lat && c.lng);
    const validS = stations.filter(s => s.lat && s.lng);
    const all = [
      ...validC.map(c => [c.lat!, c.lng!] as [number, number]),
      ...validS.map(s => [s.lat!, s.lng!] as [number, number])
    ];

    if (all.length === 0) return;
    if (all.length === 1) {
      map.setView(all[0], 16);
    } else {
      const bounds = L.latLngBounds(all);
      map.fitBounds(bounds, { padding: [60, 60] });
    }
  }, [cameras, stations, map]);
  return null;
}

function AutoPanToAlert({ cameras, alertCamIds }: { cameras: CameraEntry[], alertCamIds: Set<string> }) {
  const map = useMap();
  useEffect(() => {
    if (alertCamIds.size > 0) {
      const activeAlertCamId = Array.from(alertCamIds)[0];
      const alertCam = cameras.find(c => c.id === activeAlertCamId);
      if (alertCam && alertCam.lat && alertCam.lng) {
        map.flyTo([alertCam.lat, alertCam.lng], 16, { animate: true, duration: 1.2 });
      }
    }
  }, [alertCamIds, cameras, map]);
  return null;
}

interface Props {
  cameras: CameraEntry[];
  stations: PoliceStation[];
  alertCamIds: Set<string>;
  showToast: (msg: string, danger?: boolean) => void;
  onMarkerClick?: (id: string) => void;
}

export default function MapClient({ cameras, stations, alertCamIds, showToast, onMarkerClick }: Props) {
  const validCams = cameras.filter(c => c.lat && c.lng);
  const defaultCenter: [number, number] = validCams.length > 0
    ? [validCams[0].lat!, validCams[0].lng!]
    : [19.0760, 72.8777];

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes sonar-pulse {
          0% { transform: scale(0.5); opacity: 1; }
          100% { transform: scale(1.6); opacity: 0; }
        }
        @keyframes spin {
          100% { transform: rotate(360deg); }
        }
        .leaflet-marker-icon {
          transition: transform 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
      `}} />
      <MapContainer center={defaultCenter} zoom={15} style={{ width: "100%", height: "100%" }} zoomControl={true}>
        <TileLayer
          attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FitBounds cameras={validCams} stations={stations}/>
        <AutoPanToAlert cameras={validCams} alertCamIds={alertCamIds}/>

        {/* Render Stations */}
        {stations.filter(s => s.lat && s.lng).map(station => (
          <Marker key={station.id} position={[station.lat!, station.lng!]} icon={iconStation}>
            <Popup>
              <div style={{ fontFamily: "monospace", minWidth: 160 }}>
                <div style={{ fontWeight: 700, marginBottom: 4, fontSize: 13, color: "#00a374" }}>POLICE STATION</div>
                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>{station.name}</div>
                <div style={{ fontSize: 10, color: "#555" }}>Area: {station.location}</div>
                <div style={{ fontSize: 10, color: "#777", marginTop: 4 }}>
                  {station.lat?.toFixed(6)}°N, {station.lng?.toFixed(6)}°E
                </div>
                <div style={{ marginTop: 6, color: "var(--safe)", fontSize: 10, fontWeight: 700 }}>● ACTIVE COMMAND NODE</div>
              </div>
            </Popup>
          </Marker>
        ))}

        {/* Render Cameras */}
        {validCams.map(cam => {
          const isAlert = alertCamIds.has(cam.id);
          const iconColor = isAlert ? "#ff2244" : "#ffaa00";
          const icon = makeIcon(iconColor, isAlert);
          
          return (
            <Marker
              key={cam.id}
              position={[cam.lat!, cam.lng!]}
              icon={icon}
              eventHandlers={{ click: () => onMarkerClick?.(cam.id) }}
            >
              <Popup>
                <div style={{ fontFamily: "monospace", minWidth: 180 }}>
                  <div style={{ fontWeight: 700, marginBottom: 4, fontSize: 13 }}>{cam.label}</div>
                  <div style={{ fontSize: 11, color: "#555", marginBottom: 2 }}>
                    Type: <strong style={{ color: "#cc7700" }}>CCTV</strong>
                  </div>
                  <div style={{ fontSize: 10, color: "#777" }}>{cam.lat?.toFixed(6)}°N, {cam.lng?.toFixed(6)}°E</div>
                  {isAlert && <div style={{ marginTop: 6, color: "#cc0022", fontSize: 11, fontWeight: 700 }}>⚠ ALERT ACTIVE</div>}
                  {cam.url && <div style={{ fontSize: 10, color: "#777", marginTop: 4, wordBreak: "break-all" }}>Stream: {cam.url}</div>}
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}
