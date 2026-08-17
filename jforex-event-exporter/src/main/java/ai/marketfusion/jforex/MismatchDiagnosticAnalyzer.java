package ai.marketfusion.jforex;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.TreeMap;

/** Pure evidence analysis for targeted native-BID versus tick reconstruction diagnostics. */
final class MismatchDiagnosticAnalyzer {
    static final double PRICE_TOLERANCE = 1.0e-8;

    enum Classification {
        RECONSTRUCTION_BUG,
        PROVIDER_NATIVE_TICK_DISAGREEMENT,
        TICK_HISTORY_HOLE,
        INVALID_TICK_FILTERING,
        MINUTE_BOUNDARY_ISSUE,
        UNRESOLVED
    }

    static final class NativeBidBar {
        final double open;
        final double high;
        final double low;
        final double close;

        NativeBidBar(double open, double high, double low, double close) {
            this.open = open;
            this.high = high;
            this.low = low;
            this.close = close;
        }
    }

    static final class Issue {
        final String eventId;
        final Instant eventTime;
        final long minute;
        final String comparisonStatus;
        final NativeBidBar nativeBar;
        final TickBarReconstructor.MinuteBar rebuiltBar;
        final double openDiff;
        final double highDiff;
        final double lowDiff;
        final double closeDiff;
        final double maxDiff;
        final int tickCount;
        final int rawTickCount;
        final int invalidTickCount;
        final int duplicateTickCount;
        final boolean duplicateRemovalChangedMinute;
        final boolean duplicateRemovalChangedOhlc;
        final Classification classification;
        final String evidence;
        final List<ContextTick> context;

        private Issue(
                String eventId,
                Instant eventTime,
                long minute,
                String comparisonStatus,
                NativeBidBar nativeBar,
                TickBarReconstructor.MinuteBar rebuiltBar,
                double openDiff,
                double highDiff,
                double lowDiff,
                double closeDiff,
                double maxDiff,
                int tickCount,
                int rawTickCount,
                int invalidTickCount,
                int duplicateTickCount,
                boolean duplicateRemovalChangedMinute,
                boolean duplicateRemovalChangedOhlc,
                Classification classification,
                String evidence,
                List<ContextTick> context
        ) {
            this.eventId = eventId;
            this.eventTime = eventTime;
            this.minute = minute;
            this.comparisonStatus = comparisonStatus;
            this.nativeBar = nativeBar;
            this.rebuiltBar = rebuiltBar;
            this.openDiff = openDiff;
            this.highDiff = highDiff;
            this.lowDiff = lowDiff;
            this.closeDiff = closeDiff;
            this.maxDiff = maxDiff;
            this.tickCount = tickCount;
            this.rawTickCount = rawTickCount;
            this.invalidTickCount = invalidTickCount;
            this.duplicateTickCount = duplicateTickCount;
            this.duplicateRemovalChangedMinute = duplicateRemovalChangedMinute;
            this.duplicateRemovalChangedOhlc = duplicateRemovalChangedOhlc;
            this.classification = classification;
            this.evidence = evidence;
            this.context = context;
        }
    }

    static final class ContextTick {
        final String scope;
        final String position;
        final int rank;
        final TickBarReconstructor.Quote quote;
        final boolean accepted;
        final String rejectionReason;
        final boolean exactDuplicate;

        private ContextTick(
                String scope,
                String position,
                int rank,
                TickBarReconstructor.Quote quote,
                boolean accepted,
                String rejectionReason,
                boolean exactDuplicate
        ) {
            this.scope = scope;
            this.position = position;
            this.rank = rank;
            this.quote = quote;
            this.accepted = accepted;
            this.rejectionReason = rejectionReason;
            this.exactDuplicate = exactDuplicate;
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
            if (!(other instanceof QuoteKey)) {
                return false;
            }
            QuoteKey key = (QuoteKey) other;
            return time == key.time && bidBits == key.bidBits && askBits == key.askBits;
        }

        @Override
        public int hashCode() {
            long value = 31L * (31L * time + bidBits) + askBits;
            return (int) (value ^ (value >>> 32));
        }
    }

    private static final class Trace {
        final List<TickBarReconstructor.Quote> raw = new ArrayList<TickBarReconstructor.Quote>();
        final List<TickBarReconstructor.Quote> accepted = new ArrayList<TickBarReconstructor.Quote>();
        final Set<Long> invalidSequences = new HashSet<Long>();
        final Set<Long> duplicateSequences = new HashSet<Long>();
    }

    private MismatchDiagnosticAnalyzer() {
    }

    static List<Issue> analyze(
            String eventId,
            Instant eventTime,
            long from,
            long lastBarStart,
            TreeMap<Long, NativeBidBar> nativeBars,
            List<TickBarReconstructor.Quote> quotes
    ) {
        List<TickBarReconstructor.Quote> ordered = ordered(quotes);
        TickBarReconstructor.Result production = TickBarReconstructor.rebuild(ordered, from, lastBarStart);
        return analyzeAgainstBars(
                eventId, eventTime, from, lastBarStart, nativeBars, ordered, production.bars
        );
    }

    /** Test seam for proving that an inconsistent production replay is identified as code failure. */
    static List<Issue> analyzeAgainstBars(
            String eventId,
            Instant eventTime,
            long from,
            long lastBarStart,
            TreeMap<Long, NativeBidBar> nativeBars,
            List<TickBarReconstructor.Quote> quotes,
            TreeMap<Long, TickBarReconstructor.MinuteBar> productionBars
    ) {
        List<TickBarReconstructor.Quote> ordered = ordered(quotes);
        TreeMap<Long, Trace> traces = trace(ordered, from, lastBarStart);
        List<Issue> issues = new ArrayList<Issue>();

        for (long minute = from; minute <= lastBarStart; minute += TickBarReconstructor.MINUTE_MS) {
            NativeBidBar nativeBar = nativeBars.get(minute);
            TickBarReconstructor.MinuteBar rebuiltBar = productionBars.get(minute);
            double[] differences = differences(nativeBar, rebuiltBar);
            boolean differs = nativeBar != null && rebuiltBar != null && differences[4] > PRICE_TOLERANCE;
            if (nativeBar != null && rebuiltBar != null && !differs) {
                continue;
            }

            Trace minuteTrace = traces.containsKey(minute) ? traces.get(minute) : new Trace();
            TickBarReconstructor.MinuteBar independent = build(minuteTrace.accepted);
            boolean reconstructionAgrees = sameBidBar(rebuiltBar, independent);
            TickBarReconstructor.MinuteBar withoutDedup = build(validQuotes(minuteTrace.raw));
            boolean duplicateChangedOhlc = !sameBidOhlc(withoutDedup, independent);
            boolean boundaryMatchesNative = nativeBar != null && boundaryMatches(nativeBar, ordered, minute);

            Classification classification;
            String evidence;
            if (nativeBar == null) {
                classification = Classification.UNRESOLVED;
                evidence = "native BID M1 is absent, so native/tick agreement cannot be evaluated";
            } else if (!reconstructionAgrees) {
                classification = Classification.RECONSTRUCTION_BUG;
                evidence = "production reconstruction differs from an independent replay of the same accepted ticks";
            } else if (rebuiltBar == null) {
                if (boundaryMatchesNative) {
                    classification = Classification.MINUTE_BOUNDARY_ISSUE;
                    evidence = "no accepted in-minute ticks; exact next-boundary tick convention reproduces native BID OHLC";
                } else if (!minuteTrace.raw.isEmpty() && minuteTrace.accepted.isEmpty()
                        && minuteTrace.invalidSequences.size() == minuteTrace.raw.size()) {
                    classification = Classification.INVALID_TICK_FILTERING;
                    evidence = "provider returned ticks, but every tick failed the production validity rules";
                } else if (minuteTrace.raw.isEmpty()) {
                    classification = Classification.TICK_HISTORY_HOLE;
                    evidence = "native BID M1 exists but provider tick history returned zero ticks in the minute";
                } else {
                    classification = Classification.UNRESOLVED;
                    evidence = "rebuilt BID M1 is absent despite mixed tick evidence";
                }
            } else if (!minuteTrace.invalidSequences.isEmpty()
                    && matches(nativeBar, build(minuteTrace.raw))) {
                classification = Classification.INVALID_TICK_FILTERING;
                evidence = "including ticks rejected by the production validity rules reproduces native BID OHLC";
            } else if (boundaryMatchesNative) {
                classification = Classification.MINUTE_BOUNDARY_ISSUE;
                evidence = "an exact adjacent-minute boundary convention reproduces native BID OHLC";
            } else {
                classification = Classification.PROVIDER_NATIVE_TICK_DISAGREEMENT;
                evidence = "complete valid tick replay is internally consistent but does not reproduce native BID M1";
            }

            String status = nativeBar == null && rebuiltBar == null ? "BOTH_MISSING"
                    : nativeBar == null ? "NATIVE_BID_MISSING"
                    : rebuiltBar == null ? "REBUILT_BID_MISSING"
                    : "OHLC_MISMATCH";
            List<ContextTick> context = context(ordered, minute, minuteTrace);
            issues.add(new Issue(
                    eventId, eventTime, minute, status, nativeBar, rebuiltBar,
                    differences[0], differences[1], differences[2], differences[3], differences[4],
                    rebuiltBar == null ? 0 : rebuiltBar.tickCount,
                    minuteTrace.raw.size(), minuteTrace.invalidSequences.size(),
                    minuteTrace.duplicateSequences.size(), !minuteTrace.duplicateSequences.isEmpty(),
                    duplicateChangedOhlc, classification, evidence, context
            ));
        }
        return issues;
    }

    private static List<TickBarReconstructor.Quote> ordered(List<TickBarReconstructor.Quote> quotes) {
        List<TickBarReconstructor.Quote> result = new ArrayList<TickBarReconstructor.Quote>(quotes);
        Collections.sort(result, new Comparator<TickBarReconstructor.Quote>() {
            @Override
            public int compare(TickBarReconstructor.Quote left, TickBarReconstructor.Quote right) {
                int time = Long.compare(left.time, right.time);
                return time != 0 ? time : Long.compare(left.sequence, right.sequence);
            }
        });
        return result;
    }

    private static TreeMap<Long, Trace> trace(
            List<TickBarReconstructor.Quote> quotes, long from, long lastBarStart
    ) {
        TreeMap<Long, Trace> traces = new TreeMap<Long, Trace>();
        Set<QuoteKey> seen = new HashSet<QuoteKey>();
        for (TickBarReconstructor.Quote quote : quotes) {
            long minute = (quote.time / TickBarReconstructor.MINUTE_MS) * TickBarReconstructor.MINUTE_MS;
            if (minute < from || minute > lastBarStart) {
                continue;
            }
            Trace item = traces.get(minute);
            if (item == null) {
                item = new Trace();
                traces.put(minute, item);
            }
            item.raw.add(quote);
            if (!valid(quote)) {
                item.invalidSequences.add(quote.sequence);
                continue;
            }
            if (!seen.add(new QuoteKey(quote))) {
                item.duplicateSequences.add(quote.sequence);
                continue;
            }
            item.accepted.add(quote);
        }
        return traces;
    }

    private static boolean valid(TickBarReconstructor.Quote quote) {
        return Double.isFinite(quote.bid) && Double.isFinite(quote.ask)
                && quote.bid > 0.0 && quote.ask > 0.0 && quote.ask >= quote.bid;
    }

    private static List<TickBarReconstructor.Quote> validQuotes(List<TickBarReconstructor.Quote> quotes) {
        List<TickBarReconstructor.Quote> valid = new ArrayList<TickBarReconstructor.Quote>();
        for (TickBarReconstructor.Quote quote : quotes) {
            if (valid(quote)) {
                valid.add(quote);
            }
        }
        return valid;
    }

    private static TickBarReconstructor.MinuteBar build(List<TickBarReconstructor.Quote> quotes) {
        if (quotes.isEmpty()) {
            return null;
        }
        TickBarReconstructor.MinuteBar result = new TickBarReconstructor.MinuteBar();
        for (TickBarReconstructor.Quote quote : quotes) {
            if (result.tickCount == 0) {
                result.bidOpen = result.bidHigh = result.bidLow = result.bidClose = quote.bid;
                result.askOpen = result.askHigh = result.askLow = result.askClose = quote.ask;
            } else {
                result.bidHigh = Math.max(result.bidHigh, quote.bid);
                result.bidLow = Math.min(result.bidLow, quote.bid);
                result.bidClose = quote.bid;
                result.askHigh = Math.max(result.askHigh, quote.ask);
                result.askLow = Math.min(result.askLow, quote.ask);
                result.askClose = quote.ask;
            }
            result.tickCount++;
        }
        return result;
    }

    private static boolean boundaryMatches(
            NativeBidBar nativeBar, List<TickBarReconstructor.Quote> ordered, long minute
    ) {
        List<TickBarReconstructor.Quote> includingNextBoundary = new ArrayList<TickBarReconstructor.Quote>();
        List<TickBarReconstructor.Quote> excludingStartBoundary = new ArrayList<TickBarReconstructor.Quote>();
        long next = minute + TickBarReconstructor.MINUTE_MS;
        for (TickBarReconstructor.Quote quote : ordered) {
            if (quote.time >= minute && quote.time < next && valid(quote)) {
                includingNextBoundary.add(quote);
                if (quote.time != minute) excludingStartBoundary.add(quote);
            } else if (quote.time == next && valid(quote)) {
                includingNextBoundary.add(quote);
            }
        }
        return matches(nativeBar, build(unique(includingNextBoundary)))
                || matches(nativeBar, build(unique(excludingStartBoundary)));
    }

    private static List<TickBarReconstructor.Quote> unique(List<TickBarReconstructor.Quote> quotes) {
        List<TickBarReconstructor.Quote> result = new ArrayList<TickBarReconstructor.Quote>();
        Set<QuoteKey> seen = new HashSet<QuoteKey>();
        for (TickBarReconstructor.Quote quote : quotes) {
            if (seen.add(new QuoteKey(quote))) {
                result.add(quote);
            }
        }
        return result;
    }

    private static List<ContextTick> context(
            List<TickBarReconstructor.Quote> ordered, long minute, Trace trace
    ) {
        List<ContextTick> result = new ArrayList<ContextTick>();
        addEnds(result, trace.raw, "MINUTE", trace, 5);
        if (trace.raw.isEmpty()) {
            List<TickBarReconstructor.Quote> previous = new ArrayList<TickBarReconstructor.Quote>();
            List<TickBarReconstructor.Quote> next = new ArrayList<TickBarReconstructor.Quote>();
            long previousMinute = minute - TickBarReconstructor.MINUTE_MS;
            long nextMinute = minute + TickBarReconstructor.MINUTE_MS;
            for (TickBarReconstructor.Quote quote : ordered) {
                long quoteMinute = (quote.time / TickBarReconstructor.MINUTE_MS) * TickBarReconstructor.MINUTE_MS;
                if (quoteMinute == previousMinute) {
                    previous.add(quote);
                } else if (quoteMinute == nextMinute) {
                    next.add(quote);
                }
            }
            addTail(result, previous, "ADJACENT_PREVIOUS", 5);
            addHead(result, next, "ADJACENT_NEXT", 5);
        }
        return result;
    }

    private static void addEnds(List<ContextTick> output, List<TickBarReconstructor.Quote> values,
                                String scope, Trace trace, int count) {
        int head = Math.min(count, values.size());
        for (int index = 0; index < head; index++) {
            output.add(contextTick(scope, "FIRST", index + 1, values.get(index), trace));
        }
        // FIRST and LAST are independent requested views. They intentionally
        // overlap when a minute contains fewer than ten ticks.
        int start = Math.max(0, values.size() - count);
        int rank = 1;
        for (int index = start; index < values.size(); index++) {
            output.add(contextTick(scope, "LAST", rank++, values.get(index), trace));
        }
    }

    private static ContextTick contextTick(String scope, String position, int rank,
                                           TickBarReconstructor.Quote quote, Trace trace) {
        boolean invalid = trace.invalidSequences.contains(quote.sequence);
        boolean duplicate = trace.duplicateSequences.contains(quote.sequence);
        return new ContextTick(scope, position, rank, quote, !invalid && !duplicate,
                invalid ? invalidReason(quote) : (duplicate ? "EXACT_DUPLICATE" : ""), duplicate);
    }

    private static void addTail(List<ContextTick> output, List<TickBarReconstructor.Quote> values,
                                String scope, int count) {
        int rank = 1;
        for (int index = Math.max(0, values.size() - count); index < values.size(); index++) {
            output.add(new ContextTick(scope, "LAST", rank++, values.get(index), valid(values.get(index)),
                    valid(values.get(index)) ? "" : invalidReason(values.get(index)), false));
        }
    }

    private static void addHead(List<ContextTick> output, List<TickBarReconstructor.Quote> values,
                                String scope, int count) {
        for (int index = 0; index < Math.min(count, values.size()); index++) {
            output.add(new ContextTick(scope, "FIRST", index + 1, values.get(index), valid(values.get(index)),
                    valid(values.get(index)) ? "" : invalidReason(values.get(index)), false));
        }
    }

    private static String invalidReason(TickBarReconstructor.Quote quote) {
        if (!Double.isFinite(quote.bid) || !Double.isFinite(quote.ask)) return "NON_FINITE";
        if (quote.bid <= 0.0 || quote.ask <= 0.0) return "NON_POSITIVE";
        if (quote.ask < quote.bid) return "NEGATIVE_SPREAD";
        return "";
    }

    private static double[] differences(NativeBidBar nativeBar, TickBarReconstructor.MinuteBar rebuilt) {
        if (nativeBar == null || rebuilt == null) {
            return new double[]{Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN};
        }
        double open = Math.abs(nativeBar.open - rebuilt.bidOpen);
        double high = Math.abs(nativeBar.high - rebuilt.bidHigh);
        double low = Math.abs(nativeBar.low - rebuilt.bidLow);
        double close = Math.abs(nativeBar.close - rebuilt.bidClose);
        return new double[]{open, high, low, close, Math.max(Math.max(open, high), Math.max(low, close))};
    }

    private static boolean matches(NativeBidBar nativeBar, TickBarReconstructor.MinuteBar other) {
        return other != null && differences(nativeBar, other)[4] <= PRICE_TOLERANCE;
    }

    private static boolean sameBidBar(TickBarReconstructor.MinuteBar left, TickBarReconstructor.MinuteBar right) {
        if (left == null || right == null) return left == right;
        return sameBidOhlc(left, right) && left.tickCount == right.tickCount;
    }

    private static boolean sameBidOhlc(TickBarReconstructor.MinuteBar left, TickBarReconstructor.MinuteBar right) {
        if (left == null || right == null) return left == right;
        return Math.abs(left.bidOpen - right.bidOpen) <= PRICE_TOLERANCE
                && Math.abs(left.bidHigh - right.bidHigh) <= PRICE_TOLERANCE
                && Math.abs(left.bidLow - right.bidLow) <= PRICE_TOLERANCE
                && Math.abs(left.bidClose - right.bidClose) <= PRICE_TOLERANCE;
    }

    static String number(double value) {
        return Double.isNaN(value) ? "" : String.format(Locale.ROOT, "%.10f", value);
    }

    static String timestamp(long epochMillis) {
        return Instant.ofEpochMilli(epochMillis).toString();
    }
}
