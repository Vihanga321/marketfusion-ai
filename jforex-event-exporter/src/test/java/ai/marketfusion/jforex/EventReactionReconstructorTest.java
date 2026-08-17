package ai.marketfusion.jforex;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class EventReactionReconstructorTest {
    private static final long MINUTE = 1_700_000_040_000L;

    private static TickBarReconstructor.Quote quote(long offset, double bid, double ask, long sequence) {
        return new TickBarReconstructor.Quote(MINUTE + offset, bid, ask, sequence);
    }

    @Test
    public void midpointAndSpreadAreAggregatedFromEachTick() {
        EventReactionReconstructor.Result result = EventReactionReconstructor.rebuild(
                Arrays.asList(
                        quote(1, 1.0000, 1.4000, 0),
                        quote(2, 1.2000, 1.2100, 1),
                        quote(3, 1.1000, 1.1200, 2)
                ), MINUTE, MINUTE
        );
        EventReactionReconstructor.MinuteBar bar = result.bars.get(MINUTE);
        assertEquals(1.2000, bar.midOpen, 0.0);
        assertEquals(1.2050, bar.midHigh, 1.0e-12);
        assertEquals(1.1100, bar.midClose, 1.0e-12);
        assertNotEquals((bar.bidHigh + bar.askHigh) / 2.0, bar.midHigh, 1.0e-12);
        assertEquals(0.4000, bar.spreadOpen, 1.0e-12);
        assertEquals(0.4000, bar.spreadHigh, 1.0e-12);
        assertEquals(0.0100, bar.spreadLow, 1.0e-12);
        assertEquals(0.0200, bar.spreadClose, 1.0e-12);
    }

    @Test
    public void negativeSpreadIsRejectedAndCounted() {
        EventReactionReconstructor.Result result = EventReactionReconstructor.rebuild(
                Collections.singletonList(quote(1, 1.1000, 1.0999, 0)), MINUTE, MINUTE
        );
        assertEquals(0, result.bars.size());
        assertEquals(1, result.invalidQuoteTicks);
        assertEquals(1, result.negativeSpreadTicks);
    }

    @Test
    public void exactDuplicateRemovalPreservesProviderOrder() {
        EventReactionReconstructor.Result result = EventReactionReconstructor.rebuild(
                Arrays.asList(
                        quote(2, 1.1002, 1.1004, 2),
                        quote(1, 1.1000, 1.1002, 0),
                        quote(1, 1.1000, 1.1002, 1)
                ), MINUTE, MINUTE
        );
        EventReactionReconstructor.MinuteBar bar = result.bars.get(MINUTE);
        assertEquals(2, bar.tickCount);
        assertEquals(1, result.duplicateTicks);
        assertEquals(1.1000, bar.bidOpen, 0.0);
        assertEquals(1.1002, bar.bidClose, 0.0);
    }

    @Test
    public void pipConstantIsEurUsdPipNotPipette() {
        assertEquals(0.0001, EventReactionReconstructor.PIP_SIZE, 0.0);
    }
}
