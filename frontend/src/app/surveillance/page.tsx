"use client";
/**
 * /surveillance — redirect page
 * ================================
 * The Suraksha Drishti surveillance dashboard lives at the root route (/).
 * This page simply redirects any visit to /surveillance back to /.
 * This resolves the "Module not found: Can't resolve './page.tsx'" error
 * caused by the empty surveillance/ directory.
 */

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function SurveillanceRedirect() {
  const router = useRouter();

  useEffect(() => {
    // Immediately redirect to the main dashboard
    router.replace("/");
  }, [router]);

  // Minimal loading state shown during redirect (usually instant)
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        background: "var(--bg, #050810)",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div
        style={{
          fontFamily: "Orbitron, monospace",
          fontSize: 18,
          fontWeight: 900,
          letterSpacing: 4,
          color: "#00aaff",
          textShadow: "0 0 20px rgba(0,170,255,0.5)",
        }}
      >
        ⬡ SURAKSHADRISHTI
      </div>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          letterSpacing: 3,
          color: "rgba(255,255,255,0.3)",
          textTransform: "uppercase",
        }}
      >
        Redirecting to dashboard...
      </div>
    </div>
  );
}
