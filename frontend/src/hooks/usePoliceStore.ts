"use client";
import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";

export interface PoliceStation {
  id: string;
  name: string;
  location: string;
  password: string;
  role: "Police";
  lat?: number;
  lng?: number;
}

const TABLE_NAME = "police_stations";
const LOCAL_STORAGE_KEY = "sd_police_stations";

const SEED_ACCOUNTS: PoliceStation[] = [
  { id: "ST-01", name: "Mumbai Headquarters", location: "South Mumbai", password: "adminpassword", role: "Police", lat: 18.96, lng: 72.82 },
  { id: "ST-02", name: "Bandra Police Station", location: "West Bandra", password: "bandrapassword", role: "Police", lat: 19.05, lng: 72.83 },
  { id: "ST-03", name: "Delhi Central Division", location: "Central Delhi", password: "delhipassword", role: "Police", lat: 28.61, lng: 77.23 },
];

export function usePoliceStore() {
  const [accounts, setAccounts] = useState<PoliceStation[]>([]);
  const [loading, setLoading] = useState(true);

  // 1. Fetch and Sync State
  useEffect(() => {
    // Immediate load from localStorage to keep the UI fast and operational offline
    let cached: PoliceStation[] = [];
    try {
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored) {
        cached = JSON.parse(stored);
      } else {
        cached = SEED_ACCOUNTS;
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(SEED_ACCOUNTS));
      }
      setAccounts(cached);
    } catch (e) {
      console.warn("Failed to load local police stations cache:", e);
      cached = SEED_ACCOUNTS;
      setAccounts(SEED_ACCOUNTS);
    } finally {
      setLoading(false);
    }

    async function syncStations() {
      try {
        // Attempt network fetch from Supabase
        const { data, error } = await supabase
          .from(TABLE_NAME)
          .select("*")
          .order("name", { ascending: true });

        if (error) {
          console.warn("[Supabase Sync] Failed to read database, operating in offline fallback mode:", error.message);
          return;
        }

        if (data && data.length > 0) {
          setAccounts(data as PoliceStation[]);
          localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(data));
        }
      } catch (err) {
        console.warn("[Supabase Sync] Network offline or unreachable. Loaded from local storage cache.");
      }
    }

    syncStations();

    // Optional: Real-time subscription, wrapped in try-catch to prevent crash if server unreachable
    let channel: any = null;
    try {
      channel = supabase
        .channel("police_station_changes")
        .on("postgres_changes", { event: "*", schema: "public", table: TABLE_NAME }, (payload) => {
          if (payload.eventType === "INSERT") {
            setAccounts(prev => {
              const updated = [...prev, payload.new as PoliceStation];
              localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
              return updated;
            });
          } else if (payload.eventType === "DELETE") {
            setAccounts(prev => {
              const updated = prev.filter(a => a.id !== payload.old.id);
              localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
              return updated;
            });
          } else if (payload.eventType === "UPDATE") {
            setAccounts(prev => {
              const updated = prev.map(a => a.id === payload.new.id ? (payload.new as PoliceStation) : a);
              localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
              return updated;
            });
          }
        })
        .subscribe();
    } catch (e) {
      console.warn("Supabase real-time subscription offline:", e);
    }

    return () => {
      if (channel) {
        try {
          supabase.removeChannel(channel);
        } catch (_) {}
      }
    };
  }, []);

  const addAccount = async (name: string, location: string, pass: string) => {
    const newAcc: PoliceStation = {
      id: `ST-${Date.now()}`,
      name,
      location,
      password: pass,
      role: "Police",
      lat: 19.076 + (Math.random() - 0.5) * 0.1, // Seed approximate Mumbai region lat/lng for map integration
      lng: 72.877 + (Math.random() - 0.5) * 0.1,
    };

    // 1. Update local cache immediately so the UI is responsive and operational offline
    const updatedAccounts = [...accounts, newAcc];
    setAccounts(updatedAccounts);
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updatedAccounts));
    } catch (e) {
      console.warn("Failed to persist new police station to localStorage:", e);
    }

    // 2. Synchronize to Supabase in the background
    try {
      const { error } = await supabase
        .from(TABLE_NAME)
        .insert([newAcc]);

      if (error) {
        console.warn("[Supabase Sync] Could not synchronize new station account to database:", error.message);
      }
    } catch (err: any) {
      console.warn("[Supabase Sync] Station provisioned locally. Supabase is offline:", err.message);
    }
  };

  const removeAccount = async (id: string) => {
    // 1. Update local cache immediately
    const updatedAccounts = accounts.filter(a => a.id !== id);
    setAccounts(updatedAccounts);
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updatedAccounts));
    } catch (e) {
      console.warn("Failed to update local cache during removal:", e);
    }

    // 2. Synchronize to Supabase in the background
    try {
      const { error } = await supabase
        .from(TABLE_NAME)
        .delete()
        .eq("id", id);

      if (error) {
        console.warn("[Supabase Sync] Could not synchronize account revocation to database:", error.message);
      }
    } catch (err: any) {
      console.warn("[Supabase Sync] Station revoked locally. Supabase is offline:", err.message);
    }
  };

  return { accounts, addAccount, removeAccount, loading };
}
