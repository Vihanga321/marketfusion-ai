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
 * Read-only multi-event validation with resilient hour-chunk tick retrieval.
 *
 * Dukascopy stores historical tick data in hourly files. A whole-window
 * IHistory.getTicks call may return a partial window when one hourly file is
 * temporarily unavailable (for example HTTP 503). This validator requests each
 * hour independently, retries missing chunks, and distinguishes data retrieval
 * INCOMPLETE from a true reconstruction MISMATCH.
 *
 * No IEngine or trading/order APIs are used.
 */
public final class TickValidationBatchRobust {
    private static final String DEMO_JNLP = "http://platform.dukascopy.com/demo_3/jforex_3.jnlp";
    private static final String USER_ENV = "DUKASCOPY_USER";
    private static final String PASSWORD_ENV = "DUKASCOPY_PASSWORD";
    private static final Instrument INSTRUMENT = Instrument.EURUSD;
    private static final Period PERIOD = Period.ONE_MIN;
    private static final long MINUTE_MS = 60_000L;
    private static final long HOUR_MS = 60L * MINUTE_MS;
    private static final long WINDOW_BEFORE_MS = 10L * MINUTE_MS;
    private static final long WINDOW_AFTER_MS = 250L * MINUTE_MS;
    private static final long CONNECT_TIMEOUT_SECONDS = 45L;
    private static final long RUN_TIMEOUT_MINUTES = 40L;
    private static final double PRICE_TOLERANCE = 1.0e-8;
    private static final int NATIVE_ATTEMPTS = 4;
    private static final int CHUNK_ATTEMPTS = 5;
    private static final long RETRY_BASE_MS = 2_000L;

    private TickValidationBatchRobust() {
    }

    public static void main(String[] args) throws Exception {
        String username = requireEnvironment(USER_ENV);
        String password = requireEnvironment(PASSWORD_ENV);

        Path input = args.length >= 1
                ? Paths.get(args[0]).toAbsolutePath().normalize()
                : Paths.get("..", "data", "dukascopy", "tick_validation_sample.tsv").toAbsolutePath().normalize();
        Path outputDirectory = args.length >= 2
                ? Paths.get(args[1]).toAbsolutePath().normalize()
                : Paths.get("..", "data", "dukascopy", "tick_validation_robust").toAbsolutePath().normalize();

        List<EventRow> events = readEvents(input);
        if (events.isEmpty()) {
            throw new IllegalStateException("No events found in " + input);
        }

        System.out.println("MarketFusion robust Dukascopy tick validation");
        System.out.println("Events: " + events.size());
        System.out.println("Input:  " + input);
        System.out.println("Output: " + outputDirectory);
        System.out.println("Mode: read-only; hourly tick chunks + native BID M1 validation");

        final IClient client = ClientFactory.getDefaultInstance();
        final CountDownLatch connected = new CountDownLatch(1);
        client.setSystemListener(new ISystemListener() {
            @Override
            public void onStart(long processId) {
                System.out.println("Robust validation strategy started: " + processId);
            }

            @Override
            public void onStop(long processId) {
                System.out.println("Robust validation strategy stopped: " + processId);
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
            RobustStrategy strategy = new RobustStrategy(events, outputDirectory);
            long strategyId = client.startStrategy(strategy);

            if (!strategy.await(RUN_TIMEOUT_MINUTES, TimeUnit.MINUTES)) {
                client.stopStrategy(strategyId);
                throw new IllegalStateException("Timed out waiting for robust tick validation");
            }
            if (strategy.getFailure() != null) {
                throw new RuntimeException("Robust tick validation failed", strategy.getFailure());
            }

            System.out.println("ROBUST_TICK_VALIDATION_STATUS: " + strategy.getStatus());
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
            tickCount++;
        }
    }

    private static final class ChunkResult {
        private final long from;
        private final long to;
        private final int attempts;
        private final List<ITick> ticks;
        private final String error;

        private ChunkResult(long from, long to, int attempts, List<ITick> ticks, String error) {
            this.from = from;
            this.to = to;
            this.attempts = attempts;
            this.ticks = ticks;
            this.error = error;
        }
    }

    private static final class ValidationResult {
        private final EventRow event;
        private final int expectedMinutes;
        private final int nativeBars;
        private final int ticks;
        private final int rebuiltMinutes;
        private final int matchedMinutes;
        private final int mismatchedMinutes;
        private final int missingMinutes;
        private final int invalidSpreadTicks;
        private final int missingChunks;
        private final double maxDiff;
        private final String status;
        private final String error;

        private ValidationResult(
                EventRow event,
                int expectedMinutes,
                int nativeBars,
                int ticks,
                int rebuiltMinutes,
                int matchedMinutes,
                int mismatchedMinutes,
                int missingMinutes,
                int invalidSpreadTicks,
                int missingChunks,
                double maxDiff,
                String status,
                String error
        ) {
            this.event = event;
            this.expectedMinutes = expectedMinutes;
            this.nativeBars = nativeBars;
            this.ticks = ticks;
            this.rebuiltMinutes = rebuiltMinutes;
            this.matchedMinutes = matchedMinutes;
            this.mismatchedMinutes = mismatchedMinutes;
            this.missingMinutes = missingMinutes;
            this.invalidSpreadTicks = invalidSpreadTicks;
            this.missingChunks = missingChunks;
            this.maxDiff = maxDiff;
            this.status = status;
            this.error = error;
        }
    }

    private static final class RobustStrategy implements IStrategy {
        private final List<EventRow> events;
        private final Path outputDirectory;
        private final CountDownLatch done = new CountDownLatch(1);
        private volatile Throwable failure;
        private volatile String status = "FAIL";

        private RobustStrategy(List<EventRow> events, Path outputDirectory) {
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
                run(context.getHistory());
            } catch (Throwable ex) {
                failure = ex;
                if (ex instanceof JFException) {
                    throw (JFException) ex;
                }
                throw new JFException("Robust tick validation failed", ex);
            } finally {
                context.stop();
            }
        }

        private void run(IHistory history) throws Exception {
            Files.createDirectories(outputDirectory);
            Path summaryFile = outputDirectory.resolve("robust_tick_validation_summary.tsv");
            Path chunkFile = outputDirectory.resolve("robust_tick_chunk_status.tsv");

            int passCount = 0;
            int incompleteCount = 0;
            int mismatchCount = 0;
            int errorCount = 0;

            try (
                    BufferedWriter summary = Files.newBufferedWriter(summaryFile, StandardCharsets.UTF_8);
                    BufferedWriter chunks = Files.newBufferedWriter(chunkFile, StandardCharsets.UTF_8)
            ) {
                summary.write("event_id\tevent_timestamp_utc\texpected_minutes\tnative_bid_bars\thistorical_ticks"
                        + "\trebuilt_minutes\tmatched_minutes\tmismatched_minutes\tmissing_minutes"
                        + "\tinvalid_spread_ticks\tmissing_chunks\tmax_abs_ohlc_diff\tstatus\terror");
                summary.newLine();
                chunks.write("event_id\tchunk_from_utc\tchunk_to_utc\tattempts\tticks\tstatus\terror");
                chunks.newLine();

                for (int i = 0; i < events.size(); i++) {
                    EventRow event = events.get(i);
                    System.out.printf(Locale.ROOT, "[%d/%d] %s @ %s%n", i + 1, events.size(), event.eventId, event.eventTime);

                    ValidationResult result;
                    try {
                        result = validateEvent(history, event, chunks);
                    } catch (Throwable ex) {
                        result = new ValidationResult(event, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, "ERROR", safeMessage(ex));
                    }

                    writeSummary(summary, result);
                    if ("PASS".equals(result.status)) {
                        passCount++;
                    } else if ("INCOMPLETE".equals(result.status)) {
                        incompleteCount++;
                    } else if ("MISMATCH".equals(result.status)) {
                        mismatchCount++;
                    } else {
                        errorCount++;
                    }

                    System.out.printf(
                            Locale.ROOT,
                            "  status=%s native=%d ticks=%d rebuilt=%d matched=%d mismatched=%d missing=%d missingChunks=%d maxDiff=%.10f%n",
                            result.status,
                            result.nativeBars,
                            result.ticks,
                            result.rebuiltMinutes,
                            result.matchedMinutes,
                            result.mismatchedMinutes,
                            result.missingMinutes,
                            result.missingChunks,
                            result.maxDiff
                    );
                }
            }

            if (mismatchCount > 0 || errorCount > 0) {
                status = "FAIL";
            } else if (incompleteCount > 0) {
                status = "INCOMPLETE";
            } else {
                status = "PASS";
            }

            System.out.println("Validation events: " + events.size());
            System.out.println("PASS events: " + passCount);
            System.out.println("INCOMPLETE events: " + incompleteCount);
            System.out.println("MISMATCH events: " + mismatchCount);
            System.out.println("ERROR events: " + errorCount);
            System.out.println("Summary: " + summaryFile.toAbsolutePath());
            System.out.println("Chunks:  " + chunkFile.toAbsolutePath());
        }

        private ValidationResult validateEvent(IHistory history, EventRow event, BufferedWriter chunkWriter) throws Exception {
            long eventMs = event.eventTime.toEpochMilli();
            long from = history.getBarStart(PERIOD, eventMs - WINDOW_BEFORE_MS);
            long lastBarStart = history.getBarStart(PERIOD, eventMs + WINDOW_AFTER_MS);
            long tickTo = lastBarStart + MINUTE_MS - 1L;
            int expectedMinutes = (int) (((lastBarStart - from) / MINUTE_MS) + 1L);

            List<IBar> nativeBid = loadNativeBid(history, from, lastBarStart, event);
            if (nativeBid.isEmpty()) {
                return new ValidationResult(event, expectedMinutes, 0, 0, 0, 0, 0, expectedMinutes, 0, 0, 0.0,
                        "INCOMPLETE", "native BID M1 unavailable after retries");
            }

            List<ITick> allTicks = new ArrayList<ITick>();
            int missingChunks = 0;
            long cursor = from;
            while (cursor <= tickTo) {
                long nextHour = ((cursor / HOUR_MS) + 1L) * HOUR_MS;
                long chunkTo = Math.min(tickTo, nextHour - 1L);
                ChunkResult chunk = loadTickChunk(history, cursor, chunkTo, event);
                writeChunk(chunkWriter, event, chunk);
                if (chunk.ticks.isEmpty()) {
                    missingChunks++;
                } else {
                    allTicks.addAll(chunk.ticks);
                }
                cursor = chunkTo + 1L;
            }

            TreeMap<Long, MinuteBar> rebuilt = new TreeMap<Long, MinuteBar>();
            int invalidSpreadTicks = 0;
            for (ITick tick : allTicks) {
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

            int matched = 0;
            int mismatched = 0;
            int missingMinutes = 0;
            double maxDiff = 0.0;

            for (IBar nativeBar : nativeBid) {
                MinuteBar rebuiltBar = rebuilt.get(nativeBar.getTime());
                if (rebuiltBar == null) {
                    missingMinutes++;
                    continue;
                }
                double localMax = max4(
                        Math.abs(nativeBar.getOpen() - rebuiltBar.bidOpen),
                        Math.abs(nativeBar.getHigh() - rebuiltBar.bidHigh),
                        Math.abs(nativeBar.getLow() - rebuiltBar.bidLow),
                        Math.abs(nativeBar.getClose() - rebuiltBar.bidClose)
                );
                maxDiff = Math.max(maxDiff, localMax);
                if (localMax <= PRICE_TOLERANCE) {
                    matched++;
                } else {
                    mismatched++;
                }
            }

            String eventStatus;
            String error = "";
            if (mismatched > 0 || invalidSpreadTicks > 0) {
                eventStatus = "MISMATCH";
            } else if (missingChunks > 0 || nativeBid.size() != expectedMinutes || rebuilt.size() != expectedMinutes || missingMinutes > 0) {
                eventStatus = "INCOMPLETE";
                error = "historical retrieval incomplete; no OHLC mismatch observed in available overlap";
            } else if (matched == expectedMinutes) {
                eventStatus = "PASS";
            } else {
                eventStatus = "INCOMPLETE";
                error = "validation coverage incomplete";
            }

            return new ValidationResult(
                    event,
                    expectedMinutes,
                    nativeBid.size(),
                    allTicks.size(),
                    rebuilt.size(),
                    matched,
                    mismatched,
                    missingMinutes,
                    invalidSpreadTicks,
                    missingChunks,
                    maxDiff,
                    eventStatus,
                    error
            );
        }

        private List<IBar> loadNativeBid(IHistory history, long from, long to, EventRow event) throws InterruptedException {
            for (int attempt = 1; attempt <= NATIVE_ATTEMPTS; attempt++) {
                try {
                    List<IBar> bars = history.getBars(INSTRUMENT, PERIOD, OfferSide.BID, Filter.NO_FILTER, from, to);
                    if (bars != null && !bars.isEmpty()) {
                        return bars;
                    }
                } catch (JFException ex) {
                    System.err.println("Native BID attempt " + attempt + "/" + NATIVE_ATTEMPTS + " failed for "
                            + event.eventId + ": " + safeMessage(ex));
                }
                sleepBeforeRetry(attempt, NATIVE_ATTEMPTS);
            }
            return Collections.emptyList();
        }

        private ChunkResult loadTickChunk(IHistory history, long from, long to, EventRow event) throws InterruptedException {
            String lastError = "no ticks returned";
            for (int attempt = 1; attempt <= CHUNK_ATTEMPTS; attempt++) {
                try {
                    List<ITick> ticks = history.getTicks(INSTRUMENT, from, to);
                    if (ticks != null && !ticks.isEmpty()) {
                        return new ChunkResult(from, to, attempt, ticks, "");
                    }
                    lastError = "no ticks returned";
                } catch (JFException ex) {
                    lastError = safeMessage(ex);
                }

                System.err.println("Tick chunk attempt " + attempt + "/" + CHUNK_ATTEMPTS + " failed for "
                        + event.eventId + " " + Instant.ofEpochMilli(from) + " -> " + Instant.ofEpochMilli(to)
                        + ": " + lastError);
                sleepBeforeRetry(attempt, CHUNK_ATTEMPTS);
            }
            return new ChunkResult(from, to, CHUNK_ATTEMPTS, Collections.<ITick>emptyList(), lastError);
        }

        private void sleepBeforeRetry(int attempt, int maxAttempts) throws InterruptedException {
            if (attempt < maxAttempts) {
                long delay = RETRY_BASE_MS * (1L << (attempt - 1));
                Thread.sleep(delay);
            }
        }

        private void writeSummary(BufferedWriter writer, ValidationResult result) throws IOException {
            writer.write(result.event.eventId); writer.write('\t');
            writer.write(result.event.eventTime.toString()); writer.write('\t');
            writer.write(Integer.toString(result.expectedMinutes)); writer.write('\t');
            writer.write(Integer.toString(result.nativeBars)); writer.write('\t');
            writer.write(Integer.toString(result.ticks)); writer.write('\t');
            writer.write(Integer.toString(result.rebuiltMinutes)); writer.write('\t');
            writer.write(Integer.toString(result.matchedMinutes)); writer.write('\t');
            writer.write(Integer.toString(result.mismatchedMinutes)); writer.write('\t');
            writer.write(Integer.toString(result.missingMinutes)); writer.write('\t');
            writer.write(Integer.toString(result.invalidSpreadTicks)); writer.write('\t');
            writer.write(Integer.toString(result.missingChunks)); writer.write('\t');
            writer.write(Double.toString(result.maxDiff)); writer.write('\t');
            writer.write(result.status); writer.write('\t');
            writer.write(safeText(result.error));
            writer.newLine();
            writer.flush();
        }

        private void writeChunk(BufferedWriter writer, EventRow event, ChunkResult chunk) throws IOException {
            writer.write(event.eventId); writer.write('\t');
            writer.write(Instant.ofEpochMilli(chunk.from).toString()); writer.write('\t');
            writer.write(Instant.ofEpochMilli(chunk.to).toString()); writer.write('\t');
            writer.write(Integer.toString(chunk.attempts)); writer.write('\t');
            writer.write(Integer.toString(chunk.ticks.size())); writer.write('\t');
            writer.write(chunk.ticks.isEmpty() ? "MISSING" : "OK"); writer.write('\t');
            writer.write(safeText(chunk.error));
            writer.newLine();
            writer.flush();
        }

        @Override
        public void onStop() {
            done.countDown();
        }

        @Override
        public void onTick(Instrument instrument, ITick tick) {
            // Read-only validator: live ticks intentionally ignored.
        }

        @Override
        public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) {
            // Read-only validator: live bars intentionally ignored.
        }

        @Override
        public void onMessage(IMessage message) {
            // No trading messages consumed.
        }

        @Override
        public void onAccount(IAccount account) {
            // Account state irrelevant to historical validation.
        }
    }

    private static double max4(double a, double b, double c, double d) {
        return Math.max(Math.max(a, b), Math.max(c, d));
    }

    private static String safeMessage(Throwable throwable) {
        String message = throwable.getMessage();
        return message == null ? throwable.toString() : safeText(message);
    }

    private static String safeText(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ');
    }
}
