import { useEffect, useRef, useState } from "react";
import { fetchRealtimeQuote } from "../api/client";
import type { AssetId } from "../api/client";
import type { MarketSummary } from "../types/marketfusion";

export type RealtimeQuoteConnection = "CONNECTING" | "LIVE" | "STALE" | "DISCONNECTED";

export interface RealtimePartialCandle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number | null;
  tick_count?: number;
  is_complete: false;
  usage?: string;
}

export interface RealtimeQuoteSnapshot {
  connection?: string;
  freshness?: string;
  received_at_utc?: string | null;
  normalized_tick_utc?: string | null;
  bid?: number | null;
  ask?: number | null;
  mid?: number | null;
  spread_points?: number | null;
  sequence?: number;
  poll_interval_ms?: number | null;
  feed_delivery_ms?: number | null;
  snapshot_age_ms?: number | null;
  partial_m1?: RealtimePartialCandle | null;
}

function asSummary(snapshot: RealtimeQuoteSnapshot): MarketSummary | null {
  const connection = String(snapshot.connection ?? "UNAVAILABLE").toUpperCase();
  const freshness = String(snapshot.freshness ?? "UNAVAILABLE").toUpperCase();
  const live = connection === "LIVE" && freshness === "LIVE";
  return {
    status: live ? "PASS_LIVE" : connection || freshness,
    captured_at_utc: snapshot.received_at_utc ?? null,
    bid: snapshot.bid ?? null,
    ask: snapshot.ask ?? null,
    mid: snapshot.mid ?? null,
    spread_points: snapshot.spread_points ?? null,
  };
}

function connectionFor(snapshot: RealtimeQuoteSnapshot): RealtimeQuoteConnection {
  const connection = String(snapshot.connection ?? "").toUpperCase();
  const freshness = String(snapshot.freshness ?? "").toUpperCase();
  if (connection === "LIVE" && freshness === "LIVE") return "LIVE";
  if (connection === "DISCONNECTED") return "DISCONNECTED";
  if (connection === "LIVE" || connection === "STALE" || freshness === "STALE") return "STALE";
  return "DISCONNECTED";
}

function sourceKey(snapshot: RealtimeQuoteSnapshot) {
  return `${snapshot.sequence ?? "?"}|${snapshot.normalized_tick_utc ?? "?"}|${snapshot.received_at_utc ?? "?"}`;
}

export function useRealtimeQuote(symbol: AssetId, fallbackPollMs = 1000) {
  const [quote, setQuote] = useState<MarketSummary | null>(null);
  const [snapshot, setSnapshot] = useState<RealtimeQuoteSnapshot | null>(null);
  const [connection, setConnection] = useState<RealtimeQuoteConnection>("CONNECTING");
  const lastAdvanceAt = useRef(0);
  const lastSourceKey = useRef("");

  useEffect(() => {
    let active = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let pollController: AbortController | null = null;

    const accept = (payload: RealtimeQuoteSnapshot) => {
      if (!active) return;
      const key = sourceKey(payload);
      const advanced = key !== lastSourceKey.current;
      if (advanced) {
        lastSourceKey.current = key;
        lastAdvanceAt.current = Date.now();
      }
      setQuote(asSummary(payload));
      setSnapshot(payload);
      setConnection(connectionFor(payload));
    };

    const pollFallback = async () => {
      if (!active) return;
      if (typeof WebSocket !== "undefined" && socket?.readyState === WebSocket.OPEN && Date.now() - lastAdvanceAt.current < 2500) return;
      pollController?.abort();
      pollController = new AbortController();
      try {
        const { data } = await fetchRealtimeQuote(pollController.signal, symbol);
        accept(data as RealtimeQuoteSnapshot);
      } catch {
        if (active && (!lastAdvanceAt.current || Date.now() - lastAdvanceAt.current >= 3000)) setConnection("DISCONNECTED");
      }
    };

    const connect = () => {
      if (!active || typeof WebSocket === "undefined" || import.meta.env.MODE === "test") return;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/market/${symbol}`);
      socket.onopen = () => { if (active) setConnection("CONNECTING"); };
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(String(event.data)) as RealtimeQuoteSnapshot;
          accept(payload);
        } catch {
          if (active) setConnection("STALE");
        }
      };
      socket.onerror = () => {
        if (active && (!lastAdvanceAt.current || Date.now() - lastAdvanceAt.current >= 3000)) setConnection("STALE");
      };
      socket.onclose = () => {
        if (!active) return;
        if (!lastAdvanceAt.current || Date.now() - lastAdvanceAt.current >= 3000) setConnection("DISCONNECTED");
        reconnectTimer = window.setTimeout(connect, 1000);
      };
    };

    setConnection("CONNECTING");
    setQuote(null);
    setSnapshot(null);
    lastAdvanceAt.current = 0;
    lastSourceKey.current = "";
    connect();
    void pollFallback();
    const pollTimer = window.setInterval(() => void pollFallback(), fallbackPollMs);
    const staleTimer = window.setInterval(() => {
      if (!active || !lastAdvanceAt.current) return;
      const age = Date.now() - lastAdvanceAt.current;
      if (age >= 10000) setConnection("DISCONNECTED");
      else if (age >= 3000) setConnection("STALE");
    }, 500);

    return () => {
      active = false;
      socket?.close();
      pollController?.abort();
      if (reconnectTimer != null) window.clearTimeout(reconnectTimer);
      window.clearInterval(pollTimer);
      window.clearInterval(staleTimer);
    };
  }, [symbol, fallbackPollMs]);

  return { quote, snapshot, connection };
}
