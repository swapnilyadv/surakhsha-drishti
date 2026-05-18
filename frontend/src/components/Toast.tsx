"use client";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

interface ToastState {
  msg: string;
  danger: boolean;
  id: number;
}

let toastId = 0;

export function useToast() {
  const [toast, setToast] = useState<ToastState | null>(null);

  function showToast(msg: string, danger = false) {
    setToast({ msg, danger, id: ++toastId });
  }

  return { toast, showToast, setToast };
}

export default function Toast({ toast }: { toast: { msg: string; danger: boolean; id: number } | null }) {
  return (
    <AnimatePresence>
      {toast && (
        <motion.div
          key={toast.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          transition={{ duration: 0.25 }}
          style={{
            position: "fixed", bottom: 24, right: 24,
            background: "var(--bg2)",
            border: "1px solid var(--border)",
            borderLeft: `3px solid ${toast.danger ? "var(--danger)" : "var(--safe)"}`,
            padding: "12px 18px",
            fontFamily: "monospace", fontSize: 11, color: "var(--text)", letterSpacing: 1,
            zIndex: 9998, maxWidth: 320,
            boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
          }}
        >
          {toast.msg}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
