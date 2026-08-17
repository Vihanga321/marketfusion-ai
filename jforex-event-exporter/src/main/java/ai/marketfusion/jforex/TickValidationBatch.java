package ai.marketfusion.jforex;

import com.dukascopy.api.Filter;
import com.dukascopy.api.IAccount;
import com.dukascopy.api.IBar;
import com.dukascopy.api.IContext;
import com.dukascopy.api.IHistory;
import com.dukascopy.api.IMessage;
import com.dukascopy.api.IStrategy;
import com.dukascopy.api.ITick;
import com.dukascopy.api.Instrument;
import com.dukascopy.api.JFException;
import com.dukascopy.api.OfferSide;
import com.dukascopy.api.Period;
import com.dukascopy.api.system.ClientFactory;
import com.dukascopy.api.system.IClient;
import com.dukascopy.api.system.ISystemListener;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Read-only multi-event validator for MarketFusion AI's Dukascopy tick fallback.
 *
 * Each event is processed independently. Historical ticks are rebuilt into BID
 * and ASK M1 OHLC bars, and the rebuilt BID side is checked against Dukascopy's
 * native BID M1 bars. No IEngine or trading/order APIs are used.
 */
public final class TickValidationBatch {
    private static final String DEMO_JNLP = "http://platform.dukascopy.com/demo_3/jforex_3.jnlp";
    private static final String USER_ENV = "DUKASCOPY_USER";
    private static final String PASSWORD_ENV = "DUKASCOPY_PASSWORD";
    private static final Instrument INSTRUMENT = Instrument.EURUSD;
    private static final Period PERIOD = Period.ONE_MIN;
    private static final long MINUTE_MS = 60_000L;
    private static final long WINDOW_BEFORE_MS = 10L * MINUTE_MS;
    private static final long WINDOW_AFTER_MS = 250L * MINUTE_MS;
    private static final long CONNECT_TIMEOUT_SECONDS = 45L;
    private static final long RUN_TIMEOUT_MINUTES = 30L;
    private static final double PRICE_TOLERANCE = 1.0e-8;
    private static final int HISTORY_ATTEMPTS = 3;
    private static final long RETRY_BASE_MS = 2_000L;

    private TickValidationBatch() {
    }

    public static void main(String[] args) throws Exception {
        String username = requireEnvironment(USER_ENV);
        String password = requireEnvironment(PASSWORD_ENV);

        Path input = args.length >= 1
                ? Paths.get(args[0]).toAbsolutePath().normalize()
                : Paths.get("..", "data", "dukascopy", "tick_validation_sample.tsv").toAbsolutePath().normalize();
        Path outputDirectory = args.length >= 2
                ? Paths.get(args[1]).toAbsolutePath().normalize()
                : Paths.get("..", "data", "dukascopy", "tick_validation_batch").toAbsolutePath().normalize();

        List<EventRow> events = readEvents(input);
        if (events.isEmpty()) {
            throw new IllegalStateException("No events found in " + input);
        }

        System.out.println("MarketFusion Dukascopy multi-year tick validation");
        System.out.println("Events: " + events.size());
        System.out.println("Input:  " + input);
        System.out.println("Output: " + outputDirectory);
        System.out.println("Mode: read-only; historical ticks + native BID M1 validation");

        final IClient client = ClientFactory.getDefaultInstance();
        final CountDownLatch connected = new CountDownLatch(1);
        client.setSystemListener(new ISystemListener() {
            @Override
            public void onStart(long processId) {
                System.out.println("Batch strategy started: " + processId);
            }

            @Override
            public void onStop(long processId) {
                System.out.println("Batch strategy stopped: " + processId);
            }

            @Override
            public void onConnect() {
                System.out.println("Connected to Dukascopy DEMO JForex3 endpoint.");
                connected.countDown();
            }

            @Override
            public void onDisconnect() {
                System.out.println("Disconnected from Dukascopy.");
            }
        });

        try {
            System.out.println("Connecting to Dukascopy...");
            client.connect(DEMO_JNLP, username, password);
            if (!connected.await(CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS) || !client.isConnected()) {
                throw new IllegalStateException("Dukascopy connection timeout");
            }

            client.setSubscribedInstruments(Collections.singleton(INSTRUMENT));
            BatchStrategy strategy = new BatchStrategy(events, outputDirectory);
            long strategyId = client.startStrategy(strategy);

            if (!strategy.await(RUN_TIMEOUT_MINUTES, TimeUnit.MINUTES)) {
                client.stopStrategy(strategyId);
                throw new IllegalStateException("Timed out waiting for multi-year tick validation");
            }
            if (strategy.getFailure() != null) {
                throw new RuntimeException("Tick validation batch failed", strategy.getFailure());
            }

            System.out.println("BATCH_TICK_VALIDATION_STATUS: " + strategy.getStatus());
        } finally {
            if (client.isConnected()) {
                client.disconnect();
            }
        }
    }

    private static String requireEnvironment(String name) {
        String value = System.getenv(name);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalStateException("Missing required environment variable: " + name);
        }
        return value;
    }

    private static List<EventRow> readEvents(Path input) throws IOException {
        if (!Files.isRegularFile(input)) {
            throw new IOException("Missing event TSV: " + input);
        }

        List<EventRow> rows = new ArrayList<EventRow>();
        try (BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
            String headerLine = reader.readLine();
            if (headerLine == null) {
                return rows;
            }
            String[] headers = headerLine.split("\\t", -1);
            Map<String, Integer> index = new HashMap<String, Integer>();
            for (int i = 0; i < headers.length; i++) {
                index.put(headers[i], i);
            }
            requireColumn(index, "event_id");
            requireColumn(index, "event_type");
            requireColumn(index, "reference_period");
            requireColumn(index, "event_timestamp_utc");

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] values = line.split("\\t", -1);
                rows.add(new EventRow(
                        value(values, index, "event_id"),
                        value(values, index, "event_type"),
                        value(values, index, "reference_period"),
                        Instant.parse(value(values, index, "event_timestamp_utc"))
                ));
            }
        }
        return rows;
    }

    private static void requireColumn(Map<String, Integer> index, String name) {
        if (!index.containsKey(name)) {
            throw new IllegalArgumentException("Missing required TSV column: " + name);
        }
    }

    private static String value(String[] values, Map<String, Integer> index, String name) {
        int position = index.get(name);
        if (position >= values.length) {
            throw new IllegalArgumentException("Missing value for column " + name);
        }
        String result = values[position].trim();
        if (result.isEmpty()) {
            throw new IllegalArgumentException("Empty value for column " + name);
        }
        return result;
    }

    private static final class EventRow {
        private final String eventId;
        private final String eventType;
        private final String referencePeriod;
        private final Instant eventTime;

        private EventRow(String eventId, String eventType, String referencePeriod, Instant eventTime) {
            this.eventId = eventId;
            this.eventType = eventType;
            this.referencePeriod = referencePeriod;
            this.eventTime = eventTime;
        }
    }

    private static final class MinuteBar {
        private final long time;
        private int tickCount;
        private double bidOpen;
        private double bidHigh;
        private double bidLow;
        private double bidClose;
        private double askOpen;
        private double askHigh;
        private double askLow;
        private double askClose;
        private double bidVolumeSum;
        private double askVolumeSum;

        private MinuteBar(long time) {
            this.time = time;
        }

        private void add(ITick tick) {
            double bid = tick.getBid();
            double ask = tick.getAsk();
            if (tickCount == 0) {
                bidOpen = bidHigh = bidLow = bidClose = bid;
                askOpen = askHigh = askLow = askClose = ask;
            } else {
                bidHigh = Math.max(bidHigh, bid);
                bidLow = Math.min(bidLow, bid);
                bidClose = bid;
                askHigh = Math.max(askHigh, ask);
                askLow = Math.min(askLow, ask);
                askClose = ask;
            }
            bidVolumeSum += tick.getBidVolume();
            askVolumeSum += tick.getAskVolume();
            tickCount++;
        }
    }

    private static final class ValidationResult {
        private final EventRow event;
        private final int expectedMinutes;
        private final int nativeBidBars;
        private final int historicalTicks;
        private final int rebuiltMinutes;
        private final int overlapMinutes;
        private final int matchedMinutes;
        private final int mismatchedMinutes;
        private final int missingRebuiltMinutes;
        private final int invalidSpreadTicks;
        private final double maxAbsDiff;
        private final String status;
        private final String error;
        private final TreeMap<Long, MinuteBar> rebuilt;

        private ValidationResult(
                EventRow event,
                int expectedMinutes,
                int nativeBidBars,
                int historicalTicks,
                int rebuiltMinutes,
                int overlapMinutes,
                int matchedMinutes,
                int mismatchedMinutes,
                int missingRebuiltMinutes,
                int invalidSpreadTicks,
                double maxAbsDiff,
                String status,
                String error,
                TreeMap<Long, MinuteBar> rebuilt
        ) {
            this.event = event;
            this.expectedMinutes = expectedMinutes;
            this.nativeBidBars = nativeBidBars;
            this.historicalTicks = historicalTicks;
            this.rebuiltMinutes = rebuiltMinutes;
            this.overlapMinutes = overlapMinutes;
            this.matchedMinutes = matchedMinutes;
            this.mismatchedMinutes = mismatchedMinutes;
            this.missingRebuiltMinutes = missingRebuiltMinutes;
            this.invalidSpreadTicks = invalidSpreadTicks;
            this.maxAbsDiff = maxAbsDiff;
            this.status = status;
            this.error = error;
            this.rebuilt = rebuilt;
        }
    }

    private static final class BatchStrategy implements IStrategy {
        private final List<EventRow> events;
        private final Path outputDirectory;
        private final CountDownLatch done = new CountDownLatch(1);
        private volatile Throwable failure;
        private volatile String status = "FAIL";

        private BatchStrategy(List<EventRow> events, Path outputDirectory) {
            this.events = events;
            this.outputDirectory = outputDirectory;
        }

        private boolean await(long timeout, TimeUnit unit) throws InterruptedException {
            return done.await(timeout, unit);
        }

        private Throwable getFailure() {
            return failure;
        }

        private String getStatus() {
            return status;
        }

        @Override
        public void onStart(IContext context) throws JFException {
            try {
                context.setSubscribedInstruments(Collections.singleton(INSTRUMENT), true);
                runBatch(context.getHistory());
            } catch (Throwable ex) {
                failure = ex;
                if (ex instanceof JFException) {
                    throw (JFException) ex;
                }
                throw new JFException("Tick validation batch failed", ex);
            } finally {
                context.stop();
            }
        }

        private void runBatch(IHistory history) throws Exception {
            Files.createDirectories(outputDirectory);
            Path summaryFile = outputDirectory.resolve("tick_validation_summary.tsv");
            Path barsFile = outputDirectory.resolve("tick_validation_rebuilt_bars.tsv");

            int passCount = 0;
            int failCount = 0;

            try (
                    BufferedWriter summary = Files.newBufferedWriter(summaryFile, StandardCharsets.UTF_8);
                    BufferedWriter bars = Files.newBufferedWriter(barsFile, StandardCharsets.UTF_8)
            ) {
                summary.write("event_id\tevent_type\tevent_timestamp_utc\texpected_minutes\tnative_bid_bars\thistorical_ticks"
                        + "\trebuilt_minutes\toverlap_minutes\tmatched_minutes\tmismatched_minutes\tmissing_rebuilt_minutes"
                        + "\tinvalid_spread_ticks\tprice_tolerance\tmax_abs_ohlc_diff\tstatus\terror");
                summary.newLine();

                bars.write("event_id\tevent_type\treference_period\tevent_timestamp_utc\tbar_time_utc\ttick_count"
                        + "\tbid_open\tbid_high\tbid_low\tbid_close\task_open\task_high\task_low\task_close"
                        + "\tbid_volume_sum\task_volume_sum\tspread_open\tspread_close");
                bars.newLine();

                for (int i = 0; i < events.size(); i++) {
                    EventRow event = events.get(i);
                    System.out.printf(Locale.ROOT, "[%d/%d] %s @ %s%n", i + 1, events.size(), event.eventId, event.eventTime);

                    ValidationResult result;
                    try {
                        result = validateEvent(history, event);
                    } catch (Throwable ex) {
                        result = errorResult(event, ex);
                    }

                    writeSummary(summary, result);
                    writeBars(bars, result);

                    if ("PASS".equals(result.status)) {
                        passCount++;
                    } else {
                        failCount++;
                    }

                    System.out.printf(
                            Locale.ROOT,
                            "  status=%s native=%d ticks=%d rebuilt=%d matched=%d mismatched=%d missing=%d maxDiff=%.10f%n",
                            result.status,
                            result.nativeBidBars,
                            result.historicalTicks,
                            result.rebuiltMinutes,
                            result.matchedMinutes,
                            result.mismatchedMinutes,
                            result.missingRebuiltMinutes,
                            result.maxAbsDiff
                    );
                }
            }

            status = failCount == 0 && passCount == events.size() ? "PASS" : "FAIL";
            System.out.println("Validation events: " + events.size());
            System.out.println("PASS events: " + passCount);
            System.out.println("FAIL events: " + failCount);
            System.out.println("Summary: " + summaryFile.toAbsolutePath());
            System.out.println("Bars:    " + barsFile.toAbsolutePath());
        }

        private ValidationResult validateEvent(IHistory history, EventRow event) throws Exception {
            long eventMs = event.eventTime.toEpochMilli();
            long from = history.getBarStart(PERIOD, eventMs - WINDOW_BEFORE_MS);
            long lastBarStart = history.getBarStart(PERIOD, eventMs + WINDOW_AFTER_MS);
            long tickTo = lastBarStart + MINUTE_MS - 1L;
            int expectedMinutes = (int) (((lastBarStart - from) / MINUTE_MS) + 1L);

            List<IBar> nativeBid = loadNativeBid(history, from, lastBarStart, event);
            if (nativeBid.isEmpty()) {
                throw new JFException("Native BID M1 bars unavailable after retries");
            }

            List<ITick> ticks = loadTicks(history, from, tickTo, event);
            if (ticks.isEmpty()) {
                throw new JFException("Historical ticks unavailable after retries");
            }

            TreeMap<Long, MinuteBar> rebuilt = new TreeMap<Long, MinuteBar>();
            int invalidSpreadTicks = 0;
            for (ITick tick : ticks) {
                if (tick.getAsk() < tick.getBid()) {
                    invalidSpreadTicks++;
                }
                long minute = (tick.getTime() / MINUTE_MS) * MINUTE_MS;
                if (minute < from || minute > lastBarStart) {
                    continue;
                }
                MinuteBar bar = rebuilt.get(minute);
                if (bar == null) {
                    bar = new MinuteBar(minute);
                    rebuilt.put(minute, bar);
                }
                bar.add(tick);
            }

            int overlap = 0;
            int matched = 0;
            int mismatched = 0;
            int missingRebuilt = 0;
            double maxAbsDiff = 0.0;

            for (IBar nativeBar : nativeBid) {
                MinuteBar rebuiltBar = rebuilt.get(nativeBar.getTime());
                if (rebuiltBar == null) {
                    missingRebuilt++;
                    continue;
                }
                overlap++;
                double localMax = max4(
                        Math.abs(nativeBar.getOpen() - rebuiltBar.bidOpen),
                        Math.abs(nativeBar.getHigh() - rebuiltBar.bidHigh),
                        Math.abs(nativeBar.getLow() - rebuiltBar.bidLow),
                        Math.abs(nativeBar.getClose() - rebuiltBar.bidClose)
                );
                maxAbsDiff = Math.max(maxAbsDiff, localMax);
                if (localMax <= PRICE_TOLERANCE) {
                    matched++;
                } else {
                    mismatched++;
                }
            }

            boolean pass = nativeBid.size() == expectedMinutes
                    && rebuilt.size() == expectedMinutes
                    && overlap == expectedMinutes
                    && matched == expectedMinutes
                    && mismatched == 0
                    && missingRebuilt == 0
                    && invalidSpreadTicks == 0;

            return new ValidationResult(
                    event,
                    expectedMinutes,
                    nativeBid.size(),
                    ticks.size(),
                    rebuilt.size(),
                    overlap,
                    matched,
                    mismatched,
                    missingRebuilt,
                    invalidSpreadTicks,
                    maxAbsDiff,
                    pass ? "PASS" : "FAIL",
                    "",
                    rebuilt
            );
        }

        private List<IBar> loadNativeBid(IHistory history, long from, long to, EventRow event) throws InterruptedException {
            for (int attempt = 1; attempt <= HISTORY_ATTEMPTS; attempt++) {
                try {
                    List<IBar> bars = history.getBars(INSTRUMENT, PERIOD, OfferSide.BID, Filter.NO_FILTER, from, to);
                    if (bars != null && !bars.isEmpty()) {
                        return bars;
                    }
                } catch (JFException ex) {
                    System.err.println("Native BID attempt " + attempt + "/" + HISTORY_ATTEMPTS + " failed for "
                            + event.eventId + ": " + safeMessage(ex));
                }
                sleepBeforeRetry(attempt);
            }
            return Collections.emptyList();
        }

        private List<ITick> loadTicks(IHistory history, long from, long to, EventRow event) throws InterruptedException {
            for (int attempt = 1; attempt <= HISTORY_ATTEMPTS; attempt++) {
                try {
                    List<ITick> ticks = history.getTicks(INSTRUMENT, from, to);
                    if (ticks != null && !ticks.isEmpty()) {
                        return ticks;
                    }
                } catch (JFException ex) {
                    System.err.println("Tick attempt " + attempt + "/" + HISTORY_ATTEMPTS + " failed for "
                            + event.eventId + ": " + safeMessage(ex));
                }
                sleepBeforeRetry(attempt);
            }
            return Collections.emptyList();
        }

        private void sleepBeforeRetry(int attempt) throws InterruptedException {
            if (attempt < HISTORY_ATTEMPTS) {
                long delay = RETRY_BASE_MS * (1L << (attempt - 1));
                Thread.sleep(delay);
            }
        }

        private ValidationResult errorResult(EventRow event, Throwable ex) {
            return new ValidationResult(
                    event,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    "ERROR",
                    safeMessage(ex),
                    new TreeMap<Long, MinuteBar>()
            );
        }

        private void writeSummary(BufferedWriter writer, ValidationResult result) throws IOException {
            writer.write(result.event.eventId);
            writer.write('\t');
            writer.write(result.event.eventType);
            writer.write('\t');
            writer.write(result.event.eventTime.toString());
            writer.write('\t');
            writer.write(Integer.toString(result.expectedMinutes));
            writer.write('\t');
            writer.write(Integer.toString(result.nativeBidBars));
            writer.write('\t');
            writer.write(Integer.toString(result.historicalTicks));
            writer.write('\t');
            writer.write(Integer.toString(result.rebuiltMinutes));
            writer.write('\t');
            writer.write(Integer.toString(result.overlapMinutes));
            writer.write('\t');
            writer.write(Integer.toString(result.matchedMinutes));
            writer.write('\t');
            writer.write(Integer.toString(result.mismatchedMinutes));
            writer.write('\t');
            writer.write(Integer.toString(result.missingRebuiltMinutes));
            writer.write('\t');
            writer.write(Integer.toString(result.invalidSpreadTicks));
            writer.write('\t');
            writer.write(Double.toString(PRICE_TOLERANCE));
            writer.write('\t');
            writer.write(Double.toString(result.maxAbsDiff));
            writer.write('\t');
            writer.write(result.status);
            writer.write('\t');
            writer.write(result.error == null ? "" : result.error.replace('\t', ' ').replace('\r', ' ').replace('\n', ' '));
            writer.newLine();
            writer.flush();
        }

        private void writeBars(BufferedWriter writer, ValidationResult result) throws IOException {
            for (MinuteBar bar : result.rebuilt.values()) {
                writer.write(result.event.eventId);
                writer.write('\t');
                writer.write(result.event.eventType);
                writer.write('\t');
                writer.write(result.event.referencePeriod);
                writer.write('\t');
                writer.write(result.event.eventTime.toString());
                writer.write('\t');
                writer.write(Instant.ofEpochMilli(bar.time).toString());
                writer.write('\t');
                writer.write(Integer.toString(bar.tickCount));
                writer.write('\t');
                writer.write(Double.toString(bar.bidOpen));
                writer.write('\t');
                writer.write(Double.toString(bar.bidHigh));
                writer.write('\t');
                writer.write(Double.toString(bar.bidLow));
                writer.write('\t');
                writer.write(Double.toString(bar.bidClose));
                writer.write('\t');
                writer.write(Double.toString(bar.askOpen));
                writer.write('\t');
                writer.write(Double.toString(bar.askHigh));
                writer.write('\t');
                writer.write(Double.toString(bar.askLow));
                writer.write('\t');
                writer.write(Double.toString(bar.askClose));
                writer.write('\t');
                writer.write(Double.toString(bar.bidVolumeSum));
                writer.write('\t');
                writer.write(Double.toString(bar.askVolumeSum));
                writer.write('\t');
                writer.write(Double.toString(bar.askOpen - bar.bidOpen));
                writer.write('\t');
                writer.write(Double.toString(bar.askClose - bar.bidClose));
                writer.newLine();
            }
            writer.flush();
        }

        private static String safeMessage(Throwable throwable) {
            String message = throwable.getMessage();
            return message == null ? throwable.toString() : message;
        }

        private static double max4(double a, double b, double c, double d) {
            return Math.max(Math.max(a, b), Math.max(c, d));
        }

        @Override
        public void onStop() {
            done.countDown();
        }

        @Override
        public void onTick(Instrument instrument, ITick tick) {
            // Live ticks are intentionally ignored.
        }

        @Override
        public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) {
            // Live bars are intentionally ignored.
        }

        @Override
        public void onMessage(IMessage message) {
            // No trading messages are used.
        }

        @Override
        public void onAccount(IAccount account) {
            // Account state is irrelevant to read-only validation.
        }
    }
}