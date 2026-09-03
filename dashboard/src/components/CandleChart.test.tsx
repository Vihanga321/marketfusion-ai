import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { CandleChart } from "./CandleChart";

const closed = [
  { time: "2026-09-03T12:00:00.000Z", open: 99, high: 101, low: 98, close: 100, volume: null },
];

afterEach(() => cleanup());

describe("CandleChart display-only forming candle", () => {
  it("renders the selected timeframe forming candle green and then red as the live mid crosses its open", async () => {
    render(<CandleChart candles={closed} />);

    act(() => {
      window.dispatchEvent(new CustomEvent("marketfusion-live-quote", {
        detail: {
          connection: "LIVE",
          timeframe: "M5",
          normalizedTickUtc: "2026-09-03T12:07:15.000Z",
          mid: 101,
        },
      }));
    });

    const forming = await screen.findByLabelText("Display-only forming M5 candle");
    expect(forming.querySelector("rect")).toHaveAttribute("fill", "#20c997");

    act(() => {
      window.dispatchEvent(new CustomEvent("marketfusion-live-quote", {
        detail: {
          connection: "LIVE",
          timeframe: "M5",
          normalizedTickUtc: "2026-09-03T12:07:30.000Z",
          mid: 99,
        },
      }));
    });

    expect((await screen.findByLabelText("Display-only forming M5 candle")).querySelector("rect")).toHaveAttribute("fill", "#f0657a");
  });

  it("removes the forming candle when the live feed is no longer live", async () => {
    render(<CandleChart candles={closed} />);
    act(() => {
      window.dispatchEvent(new CustomEvent("marketfusion-live-quote", {
        detail: {
          connection: "LIVE",
          timeframe: "M15",
          normalizedTickUtc: "2026-09-03T12:17:00.000Z",
          mid: 101,
        },
      }));
    });
    expect(await screen.findByLabelText("Display-only forming M15 candle")).toBeInTheDocument();

    act(() => {
      window.dispatchEvent(new CustomEvent("marketfusion-live-quote", {
        detail: { connection: "STALE", timeframe: "M15", mid: null },
      }));
    });
    expect(screen.queryByLabelText("Display-only forming M15 candle")).not.toBeInTheDocument();
  });
});
