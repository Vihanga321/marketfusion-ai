package ai.marketfusion.jforex;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

/** Pure, deterministic UTC tick-to-M1 reconstruction used by the validator. */
final class TickBarReconstructor {
    static final long MINUTE_MS = 60_000L;

    private TickBarReconstructor() {
    }

    static final class Quote {
        final long time;
        final double bid;
        final double ask;
        final long sequence;

        Quote(long time, double bid, double ask, long sequence) {
            this.time = time;
            this.bid = bid;
            this.ask = ask;
            this.sequence = sequence;
        }
    }

    static final class MinuteBar {
        int tickCount;
        double bidOpen;
        double bidHigh;
        double bidLow;
        double bidClose;
        double askOpen;
        double askHigh;
        double askLow;
        double askClose;

        private void add(Quote quote) {
            if (tickCount == 0) {
                bidOpen = bidHigh = bidLow = bidClose = quote.bid;
                askOpen = askHigh = askLow = askClose = quote.ask;
            } else {
                bidHigh = Math.max(bidHigh, quote.bid);
                bidLow = Math.min(bidLow, quote.bid);
                bidClose = quote.bid;
                askHigh = Math.max(askHigh, quote.ask);
                askLow = Math.min(askLow, quote.ask);
                askClose = quote.ask;
            }
            tickCount++;
        }
    }

    static final class Result {
        final TreeMap<Long, MinuteBar> bars;
        final int negativeSpreadTicks;
        final int invalidQuoteTicks;
        final int duplicateTicks;

        private Result(
                TreeMap<Long, MinuteBar> bars,
                int negativeSpreadTicks,
                int invalidQuoteTicks,
                int duplicateTicks
        ) {
            this.bars = bars;
            this.negativeSpreadTicks = negativeSpreadTicks;
            this.invalidQuoteTicks = invalidQuoteTicks;
            this.duplicateTicks = duplicateTicks;
        }
    }

    private static final class QuoteKey {
        private final long time;
        private final long bidBits;
        private final long askBits;

        private QuoteKey(Quote quote) {
            this.time = quote.time;
            this.bidBits = Double.doubleToLongBits(quote.bid);
            this.askBits = Double.doubleToLongBits(quote.ask);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof QuoteKey)) {
                return false;
            }
            QuoteKey key = (QuoteKey) other;
            return time == key.time && bidBits == key.bidBits && askBits == key.askBits;
        }

        @Override
        public int hashCode() {
            long value = time;
            value = 31L * value + bidBits;
            value = 31L * value + askBits;
            return (int) (value ^ (value >>> 32));
        }
    }

    static Result rebuild(List<Quote> input, long from, long lastBarStart) {
        if (from % MINUTE_MS != 0L || lastBarStart % MINUTE_MS != 0L || lastBarStart < from) {
            throw new IllegalArgumentException("Reconstruction bounds must be ordered UTC minute starts");
        }
        List<Quote> quotes = new ArrayList<Quote>(input);
        // Collections.sort is stable. Sequence makes the intended same-timestamp
        // provider order explicit and deterministic after hourly chunks are joined.
        Collections.sort(quotes, new Comparator<Quote>() {
            @Override
            public int compare(Quote left, Quote right) {
                int timeCompare = Long.compare(left.time, right.time);
                return timeCompare != 0 ? timeCompare : Long.compare(left.sequence, right.sequence);
            }
        });

        TreeMap<Long, MinuteBar> bars = new TreeMap<Long, MinuteBar>();
        Set<QuoteKey> seen = new HashSet<QuoteKey>();
        int negativeSpreads = 0;
        int invalidQuotes = 0;
        int duplicates = 0;
        for (Quote quote : quotes) {
            boolean finite = !Double.isNaN(quote.bid) && !Double.isInfinite(quote.bid)
                    && !Double.isNaN(quote.ask) && !Double.isInfinite(quote.ask);
            if (finite && quote.ask < quote.bid) {
                negativeSpreads++;
            }
            if (!finite || quote.bid <= 0.0 || quote.ask <= 0.0 || quote.ask < quote.bid) {
                invalidQuotes++;
                continue;
            }
            QuoteKey key = new QuoteKey(quote);
            if (!seen.add(key)) {
                duplicates++;
                continue;
            }
            long minute = (quote.time / MINUTE_MS) * MINUTE_MS;
            if (minute < from || minute > lastBarStart) {
                continue;
            }
            MinuteBar bar = bars.get(minute);
            if (bar == null) {
                bar = new MinuteBar();
                bars.put(minute, bar);
            }
            bar.add(quote);
        }
        return new Result(bars, negativeSpreads, invalidQuotes, duplicates);
    }

    static long preEventBarStart(long eventTime) {
        requireMinuteAligned(eventTime);
        return eventTime - MINUTE_MS;
    }

    static long reactionBarStart(long eventTime, int horizonMinutes) {
        requireMinuteAligned(eventTime);
        if (horizonMinutes < 1) {
            throw new IllegalArgumentException("Reaction horizon must be at least one minute");
        }
        return eventTime + (horizonMinutes - 1L) * MINUTE_MS;
    }

    private static void requireMinuteAligned(long eventTime) {
        if (eventTime % MINUTE_MS != 0L) {
            throw new IllegalArgumentException("Event timestamp must be an exact UTC minute boundary");
        }
    }
}
