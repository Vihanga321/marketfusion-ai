import { useEffect, useMemo, useState } from "react";
import { Clock3 } from "lucide-react";
import { countdown, localTime } from "../utils/format";
import { getFxMarketSchedule } from "../utils/marketCalendar";

export function MarketOpenBadge() {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const schedule = useMemo(() => getFxMarketSchedule(now), [now]);

  if (schedule.isOpen || !schedule.nextOpenUtc) {
    return <div className="status-chip" title="Weekly FX reference: Sunday 17:00 America/New_York">
      <Clock3 size={12} />
      <span>FX MARKET</span>
      <b>OPEN</b>
    </div>;
  }

  return <div className="status-chip" title="Weekly FX reference: Sunday 17:00 America/New_York">
    <Clock3 size={12} />
    <span>NEXT MARKET OPEN</span>
    <b>{countdown(schedule.nextOpenUtc, now)} · {localTime(schedule.nextOpenUtc)} LKT</b>
  </div>;
}
