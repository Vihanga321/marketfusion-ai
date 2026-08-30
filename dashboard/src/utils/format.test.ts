import { describe, expect, it } from "vitest";
import { countdownState, formatColomboOpening, formatCountdown } from "./format";

const open = "2026-08-31T21:00:00Z";
const openMs = Date.parse(open);

describe("market-open countdown", () => {
  it("renders a valid closed-market countdown", () => {
    expect(countdownState(false, open, openMs - 3723000)).toBe("COUNTDOWN");
    expect(formatCountdown(3723000)).toBe("01h 02m 03s");
  });

  it("decreases deterministically with client time", () => {
    expect(formatCountdown(openMs - (openMs - 3723000))).toBe("01h 02m 03s");
    expect(formatCountdown(openMs - (openMs - 3722000))).toBe("01h 02m 02s");
  });

  it("marks opening soon under thirty minutes", () => {
    expect(formatCountdown(24 * 60 * 1000 + 13000)).toBe("24m 13s");
  });

  it("does not turn open when the local countdown expires", () => {
    expect(countdownState(false, open, openMs)).toBe("OPENING");
    expect(countdownState(false, open, openMs + 60000)).toBe("OPENING");
    expect(countdownState(true, open, openMs)).toBe("OPEN");
  });

  it("returns unknown for missing or malformed timestamps", () => {
    expect(countdownState(false, null, openMs)).toBe("UNKNOWN");
    expect(countdownState(false, "not-a-timestamp", openMs)).toBe("UNKNOWN");
    expect(formatColomboOpening(null)).toBe("UNKNOWN");
  });

  it("formats absolute opening time in Asia/Colombo", () => {
    expect(formatColomboOpening(open)).toBe("01 SEP • 02:30 AM LKT");
  });
});
