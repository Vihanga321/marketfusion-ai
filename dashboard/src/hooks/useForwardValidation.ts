import { useEffect, useState } from "react";
import type { AssetId } from "../api/client";
import type { ForwardValidationStatus } from "../types/forwardValidation";

export type ForwardValidationConnection = "LOADING" | "LIVE" | "UNAVAILABLE";

function isForwardValidationStatus(value: unknown): value is ForwardValidationStatus {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return typeof item.contract_version === "string" && typeof item.decision === "string" && !!item.contracts && typeof item.contracts === "object";
}

export function useForwardValidation(symbol: AssetId, refreshMs = 5000) {
  const [status, setStatus] = useState<ForwardValidationStatus | null>(null);
  const [connection, setConnection] = useState<ForwardValidationConnection>("LOADING");
  const [lastSuccess, setLastSuccess] = useState<Date | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    let active = true;
    let controller: AbortController | null = null;

    const load = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const response = await fetch(`/api/evaluation/forward/status?symbol=${symbol}`, {
          signal: controller.signal,
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        if (!response.ok) throw new Error(`API ${response.status}`);
        const payload: unknown = await response.json();
        if (!isForwardValidationStatus(payload)) throw new Error("Invalid forward validation payload");
        if (active) {
          setStatus(payload);
          setConnection("LIVE");
          setLastSuccess(new Date());
        }
      } catch (error) {
        if (!active || (error instanceof DOMException && error.name === "AbortError")) return;
        setConnection("UNAVAILABLE");
      }
    };

    void load();
    const timer = window.setInterval(() => void load(), refreshMs);
    return () => {
      active = false;
      controller?.abort();
      window.clearInterval(timer);
    };
  }, [symbol, refreshMs, refreshToken]);

  return {
    status,
    connection,
    lastSuccess,
    refresh: () => setRefreshToken((value) => value + 1),
  };
}
