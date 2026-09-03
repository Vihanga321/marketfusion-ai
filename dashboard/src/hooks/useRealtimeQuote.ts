import { useEffect, useRef, useState } from "react";
import { fetchRealtimeQuote } from "../api/client";
import type { AssetId } from "../api/client";
import type { MarketSummary } from "../types/marketfusion";

export type RealtimeQuoteConnection = "CONNECTING" | "LIVE" | "STALE" | "DISCONNECTED";

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
}

function asSummary(snapshot: RealtimeQuoteSnapshot): MarketSummary | null {
  if (snapshot.bid == null || snapshot.ask == null || snapshot.mid == null) return null;
  const connection = String(snapshot.connection ?? "UNAVAILABLE").toUpperCase();
  const freshness = String(snapshot.freshness ?? "UNAVAILABLE").toUpperCase();
  return {
    status: connection === "LIVE" && freshness !== "STALE" ? "PASS_LIVE" : connection,
    captured_at_utc: snapshot.received_at_utc ?? null,
    bid: snapshot.bid,
    ask: snapshot.ask,
    mid: snapshot.mid,
    spread_points: snapshot.spread_points ?? null,
  };
}

function connectionFor(snapshot: RealtimeQuoteSnapshot): RealtimeQuoteConnection {
  const connection = String(snapshot.connection ?? "").toUpperCase();
  const freshness = String(snapshot.freshness ?? "").toUpperCase();
  if (connection === "LIVE" && freshness === "LIVE") return "LIVE";
  if (connection === "LIVE" || freshness === "STALE") return "STALE";
  return "DISCONNECTED";
}

export function useRealtimeQuote(symbol: AssetId, fallbackPollMs = 1000) {
  const [quote, setQuote] = useState<MarketSummary | null>(null);
  const [snapshot, setSnapshot] = useState<RealtimeQuoteSnapshot | null>(null);
  const [connection, setConnection] = useState<RealtimeQuoteConnection>("CONNECTING");
  const lastMessageAt = useRef(0);

  useEffect(() => {
    let active = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let pollController: AbortController | null = null;

    const accept = (payload: RealtimeQuoteSnapshot) => {
      if (!active) return;
      const summary = asSummary(payload);
      if (summary) setQuote(summary);
      setSnapshot(payload);
      setConnection(connectionFor(payload));
      lastMessageAt.current = Date.now();
    };

    const pollFallback = async () => {
      if (!active) return;
      if (typeof WebSocket !== "undefined" && socket?.readyState === WebSocket.OPEN && Date.now() - lastMessageAt.current < 2500) return;
      pollController?.abort();
      pollController = new AbortController();
      try {
        const { data } = await fetchRealtimeQuote(pollController.signal, symbol);
        accept(data);
      } catch {
        if (active && Date.now() - lastMessageAt.current >= 3000) setConnection("DISCONNECTED");
      }
    };

    const connect = () => {
      if (!active || typeof WebSocket === "undefined") return;
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
      socket.onerror = () => { if (active && Date.now() - lastMessageAt.current >= 3000) setConnection("STALE"); };
      socket.onclose = () => {
        if (!active) return;
        if (Date.now() - lastMessageAt.current >= 3000) setConnection("DISCONNECTED");
        reconnectTimer = window.setTimeout(connect, 1000);
      };
    };

    setConnection("CONNECTING");
    setQuote(null);
    setSnapshot(null);
    lastMessageAt.current = 0;
    connect();
    void pollFallback();
    const pollTimer = window.setInterval(() => void pollFallback(), fallbackPollMs);
    const staleTimer = window.setInterval(() => {
      if (!active || !lastMessageAt.current) return;
      const age = Date.now() - lastMessageAt.current;
      if (age >= 5000) setConnection("DISCONNECTED");
      else if (age >= 2500) setConnection("STALE");
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
