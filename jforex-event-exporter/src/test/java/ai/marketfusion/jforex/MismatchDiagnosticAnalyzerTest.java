package ai.marketfusion.jforex;

import org.junit.Test;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class MismatchDiagnosticAnalyzerTest {
    private static final long MINUTE = 1_660_000_020_000L;
    private static final Instant EVENT = Instant.ofEpochMilli(MINUTE);

    private static TickBarReconstructor.Quote quote(long offset, double bid, double ask, long sequence) {
        return new TickBarReconstructor.Quote(MINUTE + offset, bid, ask, sequence);
    }

    private static TreeMap<Long, MismatchDiagnosticAnalyzer.NativeBidBar> nativeBar(
            double open, double high, double low, double close
    ) {
        TreeMap<Long, MismatchDiagnosticAnalyzer.NativeBidBar> bars =
                new TreeMap<Long, MismatchDiagnosticAnalyzer.NativeBidBar>();
        bars.put(MINUTE, new MismatchDiagnosticAnalyzer.NativeBidBar(open, high, low, close));
        return bars;
    }

    private static MismatchDiagnosticAnalyzer.Issue onlyIssue(
            TreeMap<Long, MismatchDiagnosticAnalyzer.NativeBidBar> nativeBars,
            List<TickBarReconstructor.Quote> quotes
    ) {
        List<MismatchDiagnosticAnalyzer.Issue> issues = MismatchDiagnosticAnalyzer.analyze(
                "event", EVENT, MINUTE, MINUTE, nativeBars, quotes
        );
        assertEquals(1, issues.size());
        return issues.get(0);
    }

    @Test
    public void completeValidReplayDisagreementIsProviderDisagreement() {
        MismatchDiagnosticAnalyzer.Issue issue = onlyIssue(
                nativeBar(1.1000, 1.1005, 1.0998, 1.1004),
                Arrays.asList(quote(1, 1.1000, 1.1001, 0), quote(2, 1.1003, 1.1004, 1))
        );
        assertEquals(MismatchDiagnosticAnalyzer.Classification.PROVIDER_NATIVE_TICK_DISAGREEMENT,
                issue.classification);
        assertEquals("OHLC_MISMATCH", issue.comparisonStatus);
    }

    @Test
    public void inconsistentProductionBarIsReconstructionBug() {
        List<TickBarReconstructor.Quote> quotes = Collections.singletonList(
                quote(1, 1.1000, 1.1001, 0)
        );
        TickBarReconstructor.MinuteBar wrong = new TickBarReconstructor.MinuteBar();
        wrong.tickCount = 1;
        wrong.bidOpen = wrong.bidHigh = wrong.bidLow = wrong.bidClose = 1.2000;
        TreeMap<Long, TickBarReconstructor.MinuteBar> production =
                new TreeMap<Long, TickBarReconstructor.MinuteBar>();
        production.put(MINUTE, wrong);
        List<MismatchDiagnosticAnalyzer.Issue> issues = MismatchDiagnosticAnalyzer.analyzeAgainstBars(
                "event", EVENT, MINUTE, MINUTE,
                nativeBar(1.1000, 1.1000, 1.1000, 1.1000), quotes, production
        );
        assertEquals(1, issues.size());
        assertEquals(MismatchDiagnosticAnalyzer.Classification.RECONSTRUCTION_BUG,
                issues.get(0).classification);
    }

    @Test
    public void missingNativeReferenceIsUnresolved() {
        MismatchDiagnosticAnalyzer.Issue issue = onlyIssue(
                new TreeMap<Long, MismatchDiagnosticAnalyzer.NativeBidBar>(),
                Collections.singletonList(quote(1, 1.1000, 1.1001, 0))
        );
        assertEquals(MismatchDiagnosticAnalyzer.Classification.UNRESOLVED, issue.classification);
        assertEquals("NATIVE_BID_MISSING", issue.comparisonStatus);
    }

    @Test
    public void nativeMinuteWithNoProviderTicksIsHistoryHole() {
        MismatchDiagnosticAnalyzer.Issue issue = onlyIssue(
                nativeBar(1.1000, 1.1000, 1.1000, 1.1000), Collections.<TickBarReconstructor.Quote>emptyList()
        );
        assertEquals(MismatchDiagnosticAnalyzer.Classification.TICK_HISTORY_HOLE, issue.classification);
        assertEquals("REBUILT_BID_MISSING", issue.comparisonStatus);
        assertEquals(0, issue.rawTickCount);
    }

    @Test
    public void allInvalidTicksAreClassifiedAsFiltering() {
        MismatchDiagnosticAnalyzer.Issue issue = onlyIssue(
                nativeBar(1.1000, 1.1000, 1.1000, 1.1000),
                Collections.singletonList(quote(1, 1.1000, 1.0999, 0))
        );
        assertEquals(MismatchDiagnosticAnalyzer.Classification.INVALID_TICK_FILTERING, issue.classification);
        assertEquals(1, issue.invalidTickCount);
        assertFalse(issue.context.get(0).accepted);
        assertEquals("NEGATIVE_SPREAD", issue.context.get(0).rejectionReason);
    }

    @Test
    public void exactNextBoundaryMatchRequiresPriceEvidence() {
        MismatchDiagnosticAnalyzer.Issue issue = onlyIssue(
                nativeBar(1.1000, 1.1000, 1.1000, 1.1000),
                Collections.singletonList(quote(60_000L, 1.1000, 1.1001, 0))
        );
        assertEquals(MismatchDiagnosticAnalyzer.Classification.MINUTE_BOUNDARY_ISSUE, issue.classification);
    }

    @Test
    public void adjacentTickAloneDoesNotProveBoundaryIssue() {
        MismatchDiagnosticAnalyzer.Issue issue = onlyIssue(
                nativeBar(1.2000, 1.2000, 1.2000, 1.2000),
                Collections.singletonList(quote(60_000L, 1.1000, 1.1001, 0))
        );
        assertEquals(MismatchDiagnosticAnalyzer.Classification.TICK_HISTORY_HOLE, issue.classification);
    }

    @Test
    public void duplicateFlagAndFirstLastContextAreDeterministic() {
        List<TickBarReconstructor.Quote> quotes = new ArrayList<TickBarReconstructor.Quote>();
        for (int index = 0; index < 12; index++) {
            quotes.add(quote(index + 1, 1.1000 + index * 0.00001, 1.1001 + index * 0.00001, index));
        }
        TickBarReconstructor.Quote original = quotes.get(11);
        quotes.add(quote(12, original.bid, original.ask, 99));
        MismatchDiagnosticAnalyzer.Issue issue = onlyIssue(
                nativeBar(1.3000, 1.3000, 1.3000, 1.3000), quotes
        );
        assertEquals(1, issue.duplicateTickCount);
        assertTrue(issue.duplicateRemovalChangedMinute);
        assertFalse(issue.duplicateRemovalChangedOhlc);
        assertEquals(10, issue.context.size());
        assertEquals("FIRST", issue.context.get(0).position);
        assertEquals(0L, issue.context.get(0).quote.sequence);
        assertEquals("LAST", issue.context.get(9).position);
        assertEquals(99L, issue.context.get(9).quote.sequence);
        assertTrue(issue.context.get(9).exactDuplicate);
    }

    @Test
    public void exactMatchProducesNoDiagnosticRow() {
        List<MismatchDiagnosticAnalyzer.Issue> issues = MismatchDiagnosticAnalyzer.analyze(
                "event", EVENT, MINUTE, MINUTE,
                nativeBar(1.1000, 1.1002, 1.1000, 1.1002),
                Arrays.asList(quote(1, 1.1000, 1.1001, 0), quote(2, 1.1002, 1.1003, 1))
        );
        assertTrue(issues.isEmpty());
    }
}
