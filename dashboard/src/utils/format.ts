export const dash = "—";

export function localTime(utc?: string | null, withDate = false): string {
  if (!utc) return dash;
  const date = new Date(utc);
  if (Number.isNaN(date.getTime())) return dash;
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Colombo", hour: "2-digit", minute: "2-digit", second: withDate ? "2-digit" : undefined,
    day: withDate ? "2-digit" : undefined, month: withDate ? "short" : undefined, year: withDate ? "numeric" : undefined,
    hour12: false
  }).format(date);
}
export function utcTime(utc?: string | null): string { return utc ? new Date(utc).toISOString().replace("T", " ").replace(".000Z", "Z") : dash; }
export function number(value?: number | null, digits = 5): string { return value == null || !Number.isFinite(value) ? dash : value.toFixed(digits); }
export function probability(value?: number | null): string { return value == null || !Number.isFinite(value) ? dash : `${(value * 100).toFixed(1)}%`; }
export function label(value?: string | null): string { return value ? value.replaceAll("_", " ") : dash; }
export function countdown(target?: string | null, now = Date.now()): string {
  if (!target) return dash;
  const seconds = Math.max(0, Math.floor((new Date(target).getTime() - now) / 1000));
  if (!Number.isFinite(seconds)) return dash;
  const hours = Math.floor(seconds / 3600); const minutes = Math.floor((seconds % 3600) / 60); const secs = seconds % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}
export function duration(seconds?: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return dash;
  const total = Math.max(0, Math.floor(seconds));
  const days = Math.floor(total / 86400); const hours = Math.floor((total % 86400) / 3600); const minutes = Math.floor((total % 3600) / 60); const secs = total % 60;
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}
export function ageFrom(utc?: string | null, now = Date.now()): string {
  if (!utc) return dash;
  const timestamp = new Date(utc).getTime();
  return Number.isFinite(timestamp) ? duration((now - timestamp) / 1000) : dash;
}
export function until(target?: string | null, now = Date.now()): string {
  if (!target) return dash;
  const timestamp = new Date(target).getTime();
  return Number.isFinite(timestamp) ? duration((timestamp - now) / 1000) : dash;
}
