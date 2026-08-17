package ai.marketfusion.jforex;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

/**
 * Enriches the frozen TickBarReconstructor semantics with per-tick MID and
 * spread M1 geometry. BID/ASK output is asserted against the production replay.
 */
final class EventReactionReconstructor {
    static final double PIP_SIZE = 0.0001;

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
        double midOpen;
        double midHigh;
        double midLow;
        double midClose;
        double spreadOpen;
        double spreadHigh;
        double spreadLow;
        double spreadClose;

        void add(TickBarReconstructor.Quote quote) {
            double mid = (quote.bid + quote.ask) / 2.0;
            double spread = quote.ask - quote.bid;
            if (tickCount == 0) {
                bidOpen = bidHigh = bidLow = bidClose = quote.bid;
                askOpen = askHigh = askLow = askClose = quote.ask;
                midOpen = midHigh = midLow = midClose = mid;
                spreadOpen = spreadHigh = spreadLow = spreadClose = spread;
            } else {
                bidHigh = Math.max(bidHigh, quote.bid);
                bidLow = Math.min(bidLow, quote.bid);
                bidClose = quote.bid;
                askHigh = Math.max(askHigh, quote.ask);
                askLow = Math.min(askLow, quote.ask);
                askClose = quote.ask;
                midHigh = Math.max(midHigh, mid);
                midLow = Math.min(midLow, mid);
                midClose = mid;
                spreadHigh = Math.max(spreadHigh, spread);
                spreadLow = Math.min(spreadLow, spread);
                spreadClose = spread;
            }
            tickCount++;
        }
    }

    static final class Result {
        final TreeMap<Long, MinuteBar> bars;
        final int invalidQuoteTicks;
        final int negativeSpreadTicks;
        final int duplicateTicks;
        final int acceptedTicks;

        Result(TreeMap<Long, MinuteBar> bars, int invalidQuoteTicks, int negativeSpreadTicks,
               int duplicateTicks, int acceptedTicks) {
            this.bars = bars;
            this.invalidQuoteTicks = invalidQuoteTicks;
            this.negativeSpreadTicks = negativeSpreadTicks;
            this.duplicateTicks = duplicateTicks;
            this.acceptedTicks = acceptedTicks;
        }
    }

    private static final class QuoteKey {
        final long time;
        final long bidBits;
        final long askBits;

        QuoteKey(TickBarReconstructor.Quote quote) {
            time = quote.time;
            bidBits = Double.doubleToLongBits(quote.bid);
            askBits = Double.doubleToLongBits(quote.ask);
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof QuoteKey)) return false;
            QuoteKey key = (QuoteKey) other;
            return time == key.time && bidBits == key.bidBits && askBits == key.askBits;
        }

        @Override
        public int hashCode() {
            long value = 31L * (31L * time + bidBits) + askBits;
            return (int) (value ^ (value >>> 32));
        }
    }

    private EventReactionReconstructor() {
    }

    static Result rebuild(List<TickBarReconstructor.Quote> input, long from, long lastBarStart) {
        TickBarReconstructor.Result production = TickBarReconstructor.rebuild(input, from, lastBarStart);
        List<TickBarReconstructor.Quote> quotes = new ArrayList<TickBarReconstructor.Quote>(input);
        Collections.sort(quotes, new Comparator<TickBarReconstructor.Quote>() {
            @Override
            public int compare(TickBarReconstructor.Quote left, TickBarReconstructor.Quote right) {
                int time = Long.compare(left.time, right.time);
                return time != 0 ? time : Long.compare(left.sequence, right.sequence);
            }
        });

        TreeMap<Long, MinuteBar> bars = new TreeMap<Long, MinuteBar>();
        Set<QuoteKey> seen = new HashSet<QuoteKey>();
        int invalid = 0;
        int negative = 0;
        int duplicates = 0;
        int accepted = 0;
        for (TickBarReconstructor.Quote quote : quotes) {
            boolean finite = !Double.isNaN(quote.bid) && !Double.isInfinite(quote.bid)
                    && !Double.isNaN(quote.ask) && !Double.isInfinite(quote.ask);
            if (finite && quote.ask < quote.bid) negative++;
            if (!finite || quote.bid <= 0.0 || quote.ask <= 0.0 || quote.ask < quote.bid) {
                invalid++;
                continue;
            }
            if (!seen.add(new QuoteKey(quote))) {
                duplicates++;
                continue;
            }
            long minute = (quote.time / TickBarReconstructor.MINUTE_MS) * TickBarReconstructor.MINUTE_MS;
            if (minute < from || minute > lastBarStart) continue;
            MinuteBar bar = bars.get(minute);
            if (bar == null) {
                bar = new MinuteBar();
                bars.put(minute, bar);
            }
            bar.add(quote);
            accepted++;
        }
        assertProductionAgreement(production.bars, bars);
        if (invalid != production.invalidQuoteTicks || negative != production.negativeSpreadTicks
                || duplicates != production.duplicateTicks) {
            throw new IllegalStateException("Enriched tick filtering differs from TickBarReconstructor");
        }
        return new Result(bars, invalid, negative, duplicates, accepted);
    }

    private static void assertProductionAgreement(
            TreeMap<Long, TickBarReconstructor.MinuteBar> production,
            TreeMap<Long, MinuteBar> enriched
    ) {
        if (!production.keySet().equals(enriched.keySet())) {
            throw new IllegalStateException("Enriched minute coverage differs from TickBarReconstructor");
        }
        for (Long minute : production.keySet()) {
            TickBarReconstructor.MinuteBar expected = production.get(minute);
            MinuteBar actual = enriched.get(minute);
            if (expected.tickCount != actual.tickCount
                    || Double.compare(expected.bidOpen, actual.bidOpen) != 0
                    || Double.compare(expected.bidHigh, actual.bidHigh) != 0
                    || Double.compare(expected.bidLow, actual.bidLow) != 0
                    || Double.compare(expected.bidClose, actual.bidClose) != 0
                    || Double.compare(expected.askOpen, actual.askOpen) != 0
                    || Double.compare(expected.askHigh, actual.askHigh) != 0
                    || Double.compare(expected.askLow, actual.askLow) != 0
                    || Double.compare(expected.askClose, actual.askClose) != 0) {
                throw new IllegalStateException("Enriched BID/ASK geometry differs at minute " + minute);
            }
        }
    }
}
