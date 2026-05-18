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

export function usePoliceStore() {
  const [accounts, setAccounts] = useState<PoliceStation[]>([]);
  const [loading, setLoading] = useState(true);

  // 1. Fetch from Supabase on mount
  useEffect(() => {
    async function fetchStations() {
      try {
        setLoading(true);
        const { data, error } = await supabase
          .from(TABLE_NAME)
          .select("*")
          .order("name", { ascending: true });

        if (error) {
          console.error("Supabase fetch error:", error.message);
          return;
        }

        if (data) {
          setAccounts(data as PoliceStation[]);
        }
      } catch (err) {
        console.error("Failed to load police stations:", err);
      } finally {
        setLoading(false);
      }
    }

    fetchStations();

    // Optional: Real-time subscription
    const channel = supabase
      .channel("police_station_changes")
      .on("postgres_changes", { event: "*", schema: "public", table: TABLE_NAME }, (payload) => {
        if (payload.eventType === "INSERT") {
          setAccounts(prev => [...prev, payload.new as PoliceStation]);
        } else if (payload.eventType === "DELETE") {
          setAccounts(prev => prev.filter(a => a.id !== payload.old.id));
        } else if (payload.eventType === "UPDATE") {
          setAccounts(prev => prev.map(a => a.id === payload.new.id ? (payload.new as PoliceStation) : a));
        }
      })
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  const addAccount = async (name: string, location: string, pass: string) => {
    try {
      const newAcc = {
        id: `ST-${Date.now()}`,
        name,
        location,
        password: pass,
        role: "Police",
      };

      const { error } = await supabase
        .from(TABLE_NAME)
        .insert([newAcc]);

      if (error) throw error;
      // Note: State is updated via the real-time subscription or manually if no subscription
    } catch (err: any) {
      console.error("Error adding police station:", err.message);
      alert("Failed to create police station. Check if table 'police_stations' exists in Supabase.");
    }
  };

  const removeAccount = async (id: string) => {
    try {
      const { error } = await supabase
        .from(TABLE_NAME)
        .delete()
        .eq("id", id);

      if (error) throw error;
    } catch (err: any) {
      console.error("Error removing police station:", err.message);
    }
  };

  return { accounts, addAccount, removeAccount, loading };
}
