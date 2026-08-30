const MARKET_TIME_ZONE = "America/New_York";
const WEEKLY_OPEN_MINUTES = 17 * 60;

const weekdayIndex: Record<string, number> = {
  Sun: 0,
  Mon: 1,
  Tue: 2,
  Wed: 3,
  Thu: 4,
  Fri: 5,
  Sat: 6,
};

type ZonedParts = {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
  weekday: number;
};

export type FxMarketSchedule = {
  isOpen: boolean;
  status: "OPEN" | "WEEKEND_CLOSED";
  nextOpenUtc: string | null;
};

function partsAt(timestampMs: number): ZonedParts {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: MARKET_TIME_ZONE,
    weekday: "short",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date(timestampMs));

  const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
  return {
    year: Number(values.year),
    month: Number(values.month),
    day: Number(values.day),
    hour: Number(values.hour),
    minute: Number(values.minute),
    second: Number(values.second),
    weekday: weekdayIndex[values.weekday] ?? 0,
  };
}

function zonedWallTimeToUtc(year: number, month: number, day: number, hour: number, minute: number): number {
  const desiredAsUtc = Date.UTC(year, month - 1, day, hour, minute, 0);
  let guess = desiredAsUtc;

  // Resolve the America/New_York UTC offset at the target wall time. Iteration
  // keeps the weekly schedule correct across EST/EDT daylight-saving changes.
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const actual = partsAt(guess);
    const actualAsUtc = Date.UTC(actual.year, actual.month - 1, actual.day, actual.hour, actual.minute, actual.second);
    const correction = desiredAsUtc - actualAsUtc;
    guess += correction;
    if (Math.abs(correction) < 1000) break;
  }

  return guess;
}

function addDays(year: number, month: number, day: number, days: number) {
  const date = new Date(Date.UTC(year, month - 1, day));
  date.setUTCDate(date.getUTCDate() + days);
  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate(),
  };
}

export function getFxMarketSchedule(nowMs = Date.now()): FxMarketSchedule {
  const ny = partsAt(nowMs);
  const minutes = ny.hour * 60 + ny.minute + ny.second / 60;

  const isOpen =
    (ny.weekday === 0 && minutes >= WEEKLY_OPEN_MINUTES) ||
    (ny.weekday >= 1 && ny.weekday <= 4) ||
    (ny.weekday === 5 && minutes < WEEKLY_OPEN_MINUTES);

  if (isOpen) {
    return { isOpen: true, status: "OPEN", nextOpenUtc: null };
  }

  let daysUntilSunday = 0;
  if (ny.weekday === 5) daysUntilSunday = 2;
  else if (ny.weekday === 6) daysUntilSunday = 1;
  else if (ny.weekday !== 0) daysUntilSunday = (7 - ny.weekday) % 7;

  const targetDate = addDays(ny.year, ny.month, ny.day, daysUntilSunday);
  const nextOpenMs = zonedWallTimeToUtc(targetDate.year, targetDate.month, targetDate.day, 17, 0);

  return {
    isOpen: false,
    status: "WEEKEND_CLOSED",
    nextOpenUtc: new Date(nextOpenMs).toISOString(),
  };
}
