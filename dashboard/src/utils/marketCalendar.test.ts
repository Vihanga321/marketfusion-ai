import { describe, expect, it } from "vitest";
import { getFxMarketSchedule } from "./marketCalendar";

describe("FX weekly market calendar", () => {
  it("counts down to the Sunday 17:00 New York open during EDT", () => {
    const schedule = getFxMarketSchedule(Date.parse("2026-08-30T19:30:00Z"));
    expect(schedule.isOpen).toBe(false);
    expect(schedule.status).toBe("WEEKEND_CLOSED");
    expect(schedule.nextOpenUtc).toBe("2026-08-30T21:00:00.000Z");
  });

  it("uses the EST offset for winter Sunday opens", () => {
    const schedule = getFxMarketSchedule(Date.parse("2026-01-04T21:00:00Z"));
    expect(schedule.isOpen).toBe(false);
    expect(schedule.nextOpenUtc).toBe("2026-01-04T22:00:00.000Z");
  });

  it("reports the market open after the weekly Sunday open", () => {
    const schedule = getFxMarketSchedule(Date.parse("2026-08-30T21:05:00Z"));
    expect(schedule.isOpen).toBe(true);
    expect(schedule.status).toBe("OPEN");
    expect(schedule.nextOpenUtc).toBeNull();
  });

  it("moves Friday after close to the following Sunday open", () => {
    const schedule = getFxMarketSchedule(Date.parse("2026-08-28T22:00:00Z"));
    expect(schedule.isOpen).toBe(false);
    expect(schedule.nextOpenUtc).toBe("2026-08-30T21:00:00.000Z");
  });
});
