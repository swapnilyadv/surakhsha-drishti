import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAppStore, Detection } from '../store/useAppStore';

// REPLACE WITH YOUR BACKEND IP (e.g. 192.168.1.5)
const BACKEND_URL = 'http://127.0.0.1:5001'; 

export function useSocket() {
  const socketRef = useRef<Socket | null>(null);
  const { setConnected, updateDetection, addIncident, setThreatLevel } = useAppStore();

  useEffect(() => {
    socketRef.current = io(BACKEND_URL);

    socketRef.current.on('connect', () => {
      console.log('Connected to AI Backend');
      setConnected(true);
    });

    socketRef.current.on('disconnect', () => {
      setConnected(false);
    });

    socketRef.current.on('detection', (data: Detection) => {
      updateDetection(data.cam_id, data);
      
      // Auto-escalate threat level based on data
      if (data.weapon || data.violence) {
        setThreatLevel('CRITICAL');
      }
    });

    return () => {
      socketRef.current?.disconnect();
    };
  }, []);

  return socketRef.current;
}
