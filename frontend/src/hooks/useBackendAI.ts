"use client";
/**
 * useBackendAI Hook
 * ==================
 * Manages WebSocket connection to FastAPI backend.
 * Receives real-time detection results:
 *   - weapon_detected, weapon_label, weapon_confidence
 *   - violence_detected, violence_label, violence_confidence
 *   - male_count, female_count
 *
 * Features:
 * - Auto-reconnect with exponential backoff
 * - Heartbeat/ping to keep connection alive
 * - Memoized state to avoid unnecessary re-renders
 * - Graceful cleanup on unmount
 */

import { useEffect, useRef, useState, useCallback } from "react";

export interface BackendDetection {
  weapon_detected: boolean;
  weapon_label: string;
  weapon_confidence: number;
  violence_detected: boolean;
  violence_label: string;
  violence_confidence: number;
  male_count: number;
  female_count: number;
  timestamp: number;
  frame_count: number;
  violence?: boolean;
  recording?: boolean;
  alarm_active?: boolean;
  event?: string;

  // Upgraded temporal metrics
  threat_level?: string;
  threat_score?: number;
  fall_detected?: boolean;
  repeated_strikes?: boolean;
  chasing_detected?: boolean;
  motion_intensity?: number;
  action?: string;
  fps?: number;
}

const DEFAULT_STATE: BackendDetection = {
  weapon_detected: false,
  weapon_label: "",
  weapon_confidence: 0,
  violence_detected: false,
  violence_label: "Non Violence",
  violence_confidence: 0,
  male_count: 0,
  female_count: 0,
  timestamp: 0,
  frame_count: 0,
  violence: false,
  recording: false,
  alarm_active: false,
  event: "NOMINAL",
  threat_level: "LOW",
  threat_score: 0.0,
  fall_detected: false,
  repeated_strikes: false,
  chasing_detected: false,
  motion_intensity: 0.0,
  action: "Normal",
  fps: 20,
};

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8765/ws/detections";
const RECONNECT_DELAY_BASE = 2000;  // 2s base
const RECONNECT_DELAY_MAX = 30000; // 30s cap
const PING_INTERVAL = 25000;       // 25s heartbeat

export function useBackendAI() {
  const [detection, setDetection] = useState<BackendDetection>(DEFAULT_STATE);
  const [connected, setConnected] = useState(false);
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectAttempts = useRef(0);
  const mountedRef = useRef(true);

  const clearTimers = useCallback(() => {
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    if (pingTimerRef.current) clearInterval(pingTimerRef.current);
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return; }
        setConnected(true);
        setBackendAvailable(true);
        reconnectAttempts.current = 0;

        // Start heartbeat ping
        pingTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, PING_INTERVAL);
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const data = JSON.parse(event.data);
          
          // Handle new evidence broadcast
          if (data.type === "new_evidence") {
            window.dispatchEvent(new CustomEvent("new-evidence-recorded", { detail: data.evidence }));
            return;
          }

          // Handle global synchronized events from WebSocket channel
          if (data.event === "dispatch_accepted") {
            window.dispatchEvent(new CustomEvent("ws-dispatch-accepted", { detail: data }));
          } else if (data.event === "need_more_help") {
            window.dispatchEvent(new CustomEvent("ws-need-more-help", { detail: data }));
          } else if (data.event === "evidence_resolved") {
            window.dispatchEvent(new CustomEvent("ws-evidence-resolved", { detail: data }));
          } else if (data.event === "violence_detected") {
            window.dispatchEvent(new CustomEvent("ws-violence-detected", { detail: data }));
          }

          if (data.type === "keepalive" || data === "pong") return;
          // Only update state if data actually changed (avoid re-renders)
          setDetection(prev => {
            const changed =
              prev.weapon_detected !== data.weapon_detected ||
              prev.violence_detected !== data.violence_detected ||
              prev.weapon_confidence !== data.weapon_confidence ||
              prev.violence_confidence !== data.violence_confidence ||
              prev.male_count !== data.male_count ||
              prev.recording !== data.recording ||
              prev.alarm_active !== data.alarm_active ||
              prev.threat_level !== data.threat_level ||
              prev.fall_detected !== data.fall_detected ||
              prev.repeated_strikes !== data.repeated_strikes ||
              prev.chasing_detected !== data.chasing_detected ||
              prev.action !== data.action ||
              prev.fps !== data.fps;
            return changed ? { ...DEFAULT_STATE, ...data } : prev;
          });
        } catch {
          // Ignore non-JSON messages (pong etc.)
        }
      };

      ws.onerror = () => {
        setBackendAvailable(false);
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnected(false);
        clearTimers();

        // Exponential backoff reconnect
        const delay = Math.min(
          RECONNECT_DELAY_BASE * Math.pow(1.5, reconnectAttempts.current),
          RECONNECT_DELAY_MAX
        );
        reconnectAttempts.current++;
        reconnectTimerRef.current = setTimeout(connect, delay);
      };

    } catch (err) {
      // WebSocket not supported / network error
      setBackendAvailable(false);
    }
  }, [clearTimers]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      clearTimers();
      wsRef.current?.close();
    };
  }, [connect, clearTimers]);

  // Poll /api/health as fallback status check
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("http://localhost:8765/api/health", { signal: AbortSignal.timeout(2000) });
        setBackendAvailable(res.ok);
      } catch {
        setBackendAvailable(false);
      }
    };
    check();
    const t = setInterval(check, 10000); // re-check every 10s
    return () => clearInterval(t);
  }, []);

  return { detection, connected, backendAvailable };
}
