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
  const anim = blinking ? 'animation: marker-pulse 1s infinite;' : '';
  const borderRadius = shape === 'circle' ? '50%' : '50% 50% 50% 0';
  const rotate = shape === 'circle' ? '0deg' : '-45deg';
  
  return L.divIcon({
    className: "",
    html: `<div style="
      width:24px;height:24px;border-radius:${borderRadius};
      background:${color};border:2px solid rgba(255,255,255,0.7);
      transform:rotate(${rotate});box-shadow:0 0 12px ${color}88;
      ${anim}
    "></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 24],
    popupAnchor: [0, -30],
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
    <MapContainer center={defaultCenter} zoom={15} style={{ width: "100%", height: "100%" }} zoomControl={true}>
      <TileLayer
        attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitBounds cameras={validCams} stations={stations}/>

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
  );
}
