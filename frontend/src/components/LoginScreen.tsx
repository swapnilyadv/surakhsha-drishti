"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { supabase } from "@/lib/supabase";

interface Props {
  onLogin: (user: string, isAdmin: boolean) => void;
}

const BOOT_STEPS = [
  { msg: "Authenticating credentials...",       result: "[OK]",    delay: 400 },
  { msg: "Loading AI inference engine...",       result: "[OK]",    delay: 700 },
  { msg: "Connecting to camera network...",      result: "[11/12]", delay: 900 },
  { msg: "Syncing incident database...",         result: "[OK]",    delay: 600 },
  { msg: "Establishing secure channel...",       result: "[OK]",    delay: 500 },
];

export default function LoginScreen({ onLogin }: Props) {
  const [email, setEmail] = useState("");
  const [pass, setPass] = useState("");
  const [error, setError] = useState(false);
  const [booting, setBooting] = useState(false);
  const [bootStep, setBootStep] = useState(0);
  const [pendingUser, setPendingUser] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);

  async function handleLogin() {
    const e = email.trim();
    const u = e.toUpperCase();
    const p = pass.trim();

    // 1. MASTER ADMIN BYPASS (High Priority - Stored in System)
    if (u === "ADMIN_001" && p === "admin@123") {
      setPendingUser("ADMIN_001");
      setIsAdmin(true);
      setBooting(true);
      return;
    }

    // 2. Try Supabase Auth (for official email-based accounts)
    if (e.includes("@")) {
      const { data, error: authError } = await supabase.auth.signInWithPassword({
        email: e,
        password: p,
      });

      if (!authError && data.user) {
        setPendingUser(data.user.email?.split('@')[0].toUpperCase() || "ADMIN");
        setIsAdmin(true); // Email logins are considered Admins in this system
        setBooting(true);
        return;
      }
    }

    // 2. Cloud Police Station Accounts (Supabase Table)
    const { data: station, error: dbError } = await supabase
      .from("police_stations")
      .select("*")
      .eq("name", e)
      .eq("password", p)
      .single();

    if (!dbError && station) {
      setPendingUser(station.name);
      setIsAdmin(false); // Station logins are not admins
      setBooting(true);
    } else {
      setError(true);
      setTimeout(() => setError(false), 3000);
    }
  }

  useEffect(() => {
    if (!booting) return;
    if (bootStep < BOOT_STEPS.length) {
      const t = setTimeout(() => setBootStep(s => s + 1), BOOT_STEPS[bootStep]?.delay ?? 500);
      return () => clearTimeout(t);
    } else {
      const t = setTimeout(() => onLogin(pendingUser, isAdmin), 800);
      return () => clearTimeout(t);
    }
  }, [booting, bootStep, pendingUser, isAdmin, onLogin]);

  const mono: React.CSSProperties = { fontFamily: "monospace" };

  return (
    <div className="fixed inset-0 flex items-center justify-center" style={{ background: "var(--bg)" }}>
      <div className="absolute inset-0" style={{ backgroundImage: "linear-gradient(rgba(0,100,200,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,100,200,0.04) 1px,transparent 1px)", backgroundSize: "40px 40px" }} />
      <div className="absolute inset-0" style={{ background: "radial-gradient(ellipse at 50% 40%,rgba(0,100,200,0.08),transparent 70%)" }} />

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
        className="bracket-box relative" style={{ width: 460, border: "1px solid var(--border-glow)", background: "rgba(8,13,26,0.97)", padding: "44px 40px 40px", boxShadow: "0 0 60px rgba(0,100,200,0.12)" }}>

        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <div style={{ width: 60, height: 60, margin: "0 auto 14px" }}>
            <svg viewBox="0 0 60 60" fill="none" style={{ filter: "drop-shadow(0 0 12px rgba(0,170,255,0.6))" }}>
              <ellipse cx="30" cy="30" rx="28" ry="18" stroke="#00aaff" strokeWidth="2"/>
              <circle cx="30" cy="30" r="10" stroke="#00aaff" strokeWidth="2"/>
              <circle cx="30" cy="30" r="5" fill="#00aaff" opacity="0.7"/>
              <circle cx="27" cy="27" r="2" fill="#fff" opacity="0.6"/>
              {[["30","2","30","10"],["30","50","30","58"],["2","30","8","30"],["52","30","58","30"]].map(([x1,y1,x2,y2],i) =>
                <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#00aaff" strokeWidth="1.5" opacity="0.5"/>)}
            </svg>
          </div>
          <h1 style={{ fontFamily: "Orbitron,sans-serif", fontSize: 20, fontWeight: 900, letterSpacing: 4, color: "var(--accent)", textShadow: "0 0 20px rgba(0,170,255,0.4)" }}>SURAKSHADRISHTI</h1>
          <p style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 3, marginTop: 4 }}>MUMBAI TRANSIT SURVEILLANCE NETWORK</p>
        </div>

        {/* System info strip */}
        <div style={{ ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 1, background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", padding: "6px 10px", marginBottom: 24, lineHeight: 1.7 }}>
          <div>NODE: MU-CSTN-07 | UPTIME: 14d 07h | CAMERAS: <span style={{ color: "var(--safe)" }}>11/12 ONLINE</span></div>
          <div>LAST INCIDENT: <span style={{ color: "var(--danger)" }}>09:14:33</span> | CLASSIFIED LEVEL 2</div>
        </div>

        {!booting ? (
          <>
            <label style={{ ...mono, fontSize: 10, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase", display: "block", marginBottom: 6 }}>Email / Operator ID</label>
            <input value={email} onChange={e => setEmail(e.target.value)}
              placeholder="admin@suraksha.gov"
              style={{ width: "100%", background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 14, padding: "11px 13px", outline: "none", marginBottom: 14 }}/>
            <label style={{ ...mono, fontSize: 10, color: "var(--text-dim)", letterSpacing: 2, textTransform: "uppercase", display: "block", marginBottom: 6 }}>Passphrase</label>
            <input type="password" value={pass} onChange={e => setPass(e.target.value)} onKeyDown={e => e.key === "Enter" && handleLogin()}
              placeholder="••••••••"
              style={{ width: "100%", background: "rgba(0,100,200,0.06)", border: "1px solid var(--border)", color: "var(--text-bright)", ...mono, fontSize: 14, padding: "11px 13px", outline: "none" }}/>
            <button onClick={handleLogin}
              style={{ width: "100%", marginTop: 24, padding: "13px", background: "var(--accent2)", border: "none", color: "#fff", fontFamily: "Orbitron,sans-serif", fontSize: 12, fontWeight: 700, letterSpacing: 4, cursor: "pointer", textTransform: "uppercase" }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "var(--accent)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "var(--accent2)"; }}>
              AUTHENTICATE
            </button>
            <AnimatePresence>
              {error && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  style={{ ...mono, fontSize: 11, color: "var(--danger)", textAlign: "center", marginTop: 12, letterSpacing: 1 }}>
                  ⚠ AUTHENTICATION FAILED — INVALID CREDENTIALS
                </motion.div>
              )}
            </AnimatePresence>
          </>
        ) : (
          /* Boot sequence */
          <div style={{ ...mono, fontSize: 12, lineHeight: 2, color: "var(--text-dim)" }}>
            {BOOT_STEPS.slice(0, bootStep).map((step, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between" }}>
                <span>{">"} {step.msg}</span>
                <span style={{ color: step.result.includes("11/12") ? "var(--warning)" : "var(--safe)" }}>{step.result}</span>
              </div>
            ))}
            {bootStep >= BOOT_STEPS.length && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                style={{ marginTop: 10, color: "var(--safe)", fontWeight: 700, letterSpacing: 2 }}>
                ACCESS GRANTED — Welcome, {pendingUser}
              </motion.div>
            )}
            {bootStep < BOOT_STEPS.length && (
              <motion.span animate={{ opacity: [1, 0, 1] }} transition={{ repeat: Infinity, duration: 0.8 }}>█</motion.span>
            )}
          </div>
        )}

        <div style={{ position: "absolute", bottom: -26, left: 0, right: 0, textAlign: "center", ...mono, fontSize: 9, color: "var(--text-dim)", letterSpacing: 3 }}>
          RESTRICTED ACCESS · AUTHORIZED PERSONNEL ONLY
        </div>
      </motion.div>
    </div>
  );
}
