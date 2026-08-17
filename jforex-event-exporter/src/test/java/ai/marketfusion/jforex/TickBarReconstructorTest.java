package ai.marketfusion.jforex;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TickBarReconstructorTest {
    private static TickBarReconstructor.Quote q(long time, double bid, double ask, long sequence) {
        return new TickBarReconstructor.Quote(time, bid, ask, sequence);
    }

    @Test
    public void reconstructsBidAndAskOhlcInChronologicalOrder() {
        List<TickBarReconstructor.Quote> quotes = Arrays.asList(
                q(40_000L, 1.1010, 1.1013, 2),
                q(1_000L, 1.1000, 1.1002, 0),
                q(20_000L, 1.1020, 1.1024, 1),
                q(59_999L, 1.0990, 1.0993, 3)
        );
        TickBarReconstructor.Result result = TickBarReconstructor.rebuild(quotes, 0L, 0L);
        TickBarReconstructor.MinuteBar bar = result.bars.get(0L);
        assertEquals(1.1000, bar.bidOpen, 0.0);
        assertEquals(1.1020, bar.bidHigh, 0.0);
        assertEquals(1.0990, bar.bidLow, 0.0);
        assertEquals(1.0990, bar.bidClose, 0.0);
        assertEquals(1.1002, bar.askOpen, 0.0);
        assertEquals(1.1024, bar.askHigh, 0.0);
        assertEquals(1.0993, bar.askLow, 0.0);
        assertEquals(1.0993, bar.askClose, 0.0);
    }

    @Test
    public void handlesMinuteBoundariesAndStableSameTimestampOrder() {
        List<TickBarReconstructor.Quote> quotes = Arrays.asList(
                q(60_000L, 1.2000, 1.2002, 2),
                q(59_999L, 1.1000, 1.1002, 0),
                q(60_000L, 1.3000, 1.3002, 3)
        );
        TickBarReconstructor.Result result = TickBarReconstructor.rebuild(quotes, 0L, 60_000L);
        assertEquals(2, result.bars.size());
        assertEquals(1.1000, result.bars.get(0L).bidClose, 0.0);
        assertEquals(1.2000, result.bars.get(60_000L).bidOpen, 0.0);
        assertEquals(1.3000, result.bars.get(60_000L).bidClose, 0.0);
    }

    @Test
    public void rejectsInvalidQuotesAndDeduplicatesExactTicks() {
        TickBarReconstructor.Quote valid = q(1_000L, 1.1000, 1.1002, 0);
        List<TickBarReconstructor.Quote> quotes = Arrays.asList(
                valid,
                q(1_000L, 1.1000, 1.1002, 1),
                q(2_000L, 1.2000, 1.1000, 2),
                q(3_000L, 0.0, 1.1000, 3),
                q(4_000L, Double.NaN, 1.1000, 4)
        );
        TickBarReconstructor.Result result = TickBarReconstructor.rebuild(quotes, 0L, 0L);
        assertEquals(1, result.bars.get(0L).tickCount);
        assertEquals(1, result.duplicateTicks);
        assertEquals(1, result.negativeSpreadTicks);
        assertEquals(3, result.invalidQuoteTicks);
    }

    @Test
    public void reactionIndexingMatchesDocumentedCloseSemantics() {
        long event = 1_800_000L;
        assertEquals(event - 60_000L, TickBarReconstructor.preEventBarStart(event));
        assertEquals(event, TickBarReconstructor.reactionBarStart(event, 1));
        assertEquals(event + 4L * 60_000L, TickBarReconstructor.reactionBarStart(event, 5));
        assertEquals(event + 14L * 60_000L, TickBarReconstructor.reactionBarStart(event, 15));
        assertEquals(event + 59L * 60_000L, TickBarReconstructor.reactionBarStart(event, 60));
        assertEquals(event + 239L * 60_000L, TickBarReconstructor.reactionBarStart(event, 240));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonMinuteAlignedEvents() {
        TickBarReconstructor.reactionBarStart(1_800_001L, 1);
    }
}
