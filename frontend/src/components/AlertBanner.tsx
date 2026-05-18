"use client";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  isAlert: boolean;
  alertText: string;
  onAck: () => void;
}

export default function AlertBanner({ isAlert, alertText, onAck }: Props) {
  return (
    <AnimatePresence>
      {isAlert && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.25 }}
          style={{
            background: "rgba(255,34,68,0.13)",
            borderBottom: "2px solid var(--danger)",
            padding: "10px 20px",
            fontFamily: "monospace",
            fontSize: 12,
            color: "var(--danger)",
            letterSpacing: 2,
            textTransform: "uppercase",
            display: "flex",
            alignItems: "center",
            gap: 16,
            animation: "alert-flash-bg 0.5s infinite",
            flexShrink: 0,
          }}
        >
          <span>⚠</span>
          <span>{alertText}</span>
          <button
            onClick={onAck}
            style={{
              marginLeft: "auto", fontFamily: "monospace", fontSize: 10,
              background: "transparent", border: "1px solid var(--danger)",
              color: "var(--danger)", padding: "4px 12px", cursor: "pointer", letterSpacing: 1,
              transition: "all 0.15s",
            }}
            onMouseEnter={e => { const b = e.target as HTMLElement; b.style.background = "var(--danger)"; b.style.color = "#fff"; }}
            onMouseLeave={e => { const b = e.target as HTMLElement; b.style.background = "transparent"; b.style.color = "var(--danger)"; }}
          >
            ACKNOWLEDGE
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
