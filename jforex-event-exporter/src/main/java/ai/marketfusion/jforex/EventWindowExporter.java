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
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Read-only EUR/USD M1 historical event-window collector for MarketFusion AI.
 *
 * This program intentionally never obtains IEngine and contains no order or
 * position-management code. Credentials are read only from process environment
 * variables and are never written to disk.
 */
public final class EventWindowExporter {
    private static final String DEMO_JNLP = "http://platform.dukascopy.com/demo_3/jforex_3.jnlp";
    private static final String USER_ENV = "DUKASCOPY_USER";
    private static final String PASSWORD_ENV = "DUKASCOPY_PASSWORD";
    private static final Instrument INSTRUMENT = Instrument.EURUSD;
    private static final Period PERIOD = Period.ONE_MIN;
    private static final long MINUTE_MS = 60_000L;
    private static final long WINDOW_BEFORE_MS = 10L * MINUTE_MS;
    private static final long WINDOW_AFTER_MS = 250L * MINUTE_MS;
    private static final long CONNECT_TIMEOUT_SECONDS = 45L;
    private static final int HISTORY_MAX_ATTEMPTS = 5;
    private static final long HISTORY_RETRY_BASE_MS = 2_000L;

    private EventWindowExporter() {
    }

    public static void main(String[] args) throws Exception {
        String username = requireEnvironment(USER_ENV);
        String password = requireEnvironment(PASSWORD_ENV);

        Path input = args.length >= 1
                ? Paths.get(args[0]).toAbsolutePath().normalize()
                : Paths.get("..", "data", "dukascopy", "events_for_jforex.tsv").toAbsolutePath().normalize();
        Path outputDirectory = args.length >= 2
                ? Paths.get(args[1]).toAbsolutePath().normalize()
                : Paths.get("..", "data", "dukascopy", "event_windows").toAbsolutePath().normalize();

        List<EventRow> events = readEvents(input);
        if (events.isEmpty()) {
            throw new IllegalStateException("No events found in " + input);
        }

        System.out.println("Read-only JForex event exporter");
        System.out.println("Events: " + events.size());
        System.out.println("Input:  " + input);
        System.out.println("Output: " + outputDirectory);
        System.out.println("Instrument: EUR/USD, period: ONE_MIN, sides: BID + ASK");

        final IClient client = ClientFactory.getDefaultInstance();
        final CountDownLatch connected = new CountDownLatch(1);
        client.setSystemListener(new ISystemListener() {
            @Override
            public void onStart(long processId) {
                System.out.println("Strategy started: " + processId);
            }

            @Override
            public void onStop(long processId) {
                System.out.println("Strategy stopped: " + processId);
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
                throw new IllegalStateException(
                        "Dukascopy client did not reach connected state within "
                                + CONNECT_TIMEOUT_SECONDS
                                + " seconds"
                );
            }

            client.setSubscribedInstruments(Collections.singleton(INSTRUMENT));

            ExportStrategy strategy = new ExportStrategy(events, outputDirectory);
            long strategyId = client.startStrategy(strategy);

            if (!strategy.await(10, TimeUnit.MINUTES)) {
                client.stopStrategy(strategyId);
                throw new IllegalStateException("Timed out waiting for event-window export to finish");
            }
            if (strategy.getFailure() != null) {
                throw new RuntimeException("JForex export failed", strategy.getFailure());
            }

            if (strategy.getMissingSideCount() == 0) {
                System.out.println("EXPORT_STATUS: SUCCESS");
            } else {
                System.out.println(
                        "EXPORT_STATUS: PARTIAL (missing event/sides="
                                + strategy.getMissingSideCount()
                                + ")"
                );
            }
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
            int lineNumber = 1;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] values = line.split("\\t", -1);
                try {
                    rows.add(new EventRow(
                            value(values, index, "event_id"),
                            value(values, index, "event_type"),
                            value(values, index, "reference_period"),
                            Instant.parse(value(values, index, "event_timestamp_utc"))
                    ));
                } catch (RuntimeException ex) {
                    throw new IOException("Invalid event TSV at line " + lineNumber + ": " + ex.getMessage(), ex);
                }
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
        String value = values[position].trim();
        if (value.isEmpty()) {
            throw new IllegalArgumentException("Empty value for column " + name);
        }
        return value;
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

    private static final class LoadResult {
        private final List<IBar> bars;
        private final int attempts;
        private final String error;

        private LoadResult(List<IBar> bars, int attempts, String error) {
            this.bars = bars;
            this.attempts = attempts;
            this.error = error;
        }
    }

    private static final class ExportStrategy implements IStrategy {
        private final List<EventRow> events;
        private final Path outputDirectory;
        private final CountDownLatch done = new CountDownLatch(1);
        private volatile Throwable failure;
        private volatile int missingSideCount;
        private IContext context;

        private ExportStrategy(List<EventRow> events, Path outputDirectory) {
            this.events = events;
            this.outputDirectory = outputDirectory;
        }

        private boolean await(long timeout, TimeUnit unit) throws InterruptedException {
            return done.await(timeout, unit);
        }

        private Throwable getFailure() {
            return failure;
        }

        private int getMissingSideCount() {
            return missingSideCount;
        }

        @Override
        public void onStart(IContext context) throws JFException {
            this.context = context;
            try {
                context.setSubscribedInstruments(Collections.singleton(INSTRUMENT), true);
                export(context.getHistory());
            } catch (Throwable ex) {
                failure = ex;
                if (ex instanceof JFException) {
                    throw (JFException) ex;
                }
                throw new JFException("Event-window export failed", ex);
            } finally {
                context.stop();
            }
        }

        private void export(IHistory history) throws Exception {
            Files.createDirectories(outputDirectory);
            Path finalFile = outputDirectory.resolve("eurusd_m1_event_windows.tsv");
            Path tempFile = outputDirectory.resolve("eurusd_m1_event_windows.tmp.tsv");
            Path finalStatusFile = outputDirectory.resolve("eurusd_m1_event_window_status.tsv");
            Path tempStatusFile = outputDirectory.resolve("eurusd_m1_event_window_status.tmp.tsv");

            long rowCount = 0L;
            try (
                    BufferedWriter writer = Files.newBufferedWriter(tempFile, StandardCharsets.UTF_8);
                    BufferedWriter statusWriter = Files.newBufferedWriter(tempStatusFile, StandardCharsets.UTF_8)
            ) {
                writer.write("event_id\tevent_type\treference_period\tevent_timestamp_utc\toffer_side\tbar_time_utc\topen\thigh\tlow\tclose\tvolume");
                writer.newLine();
                statusWriter.write("event_id\tevent_type\tevent_timestamp_utc\toffer_side\tattempts\tbars_returned\tstatus\terror");
                statusWriter.newLine();

                for (int i = 0; i < events.size(); i++) {
                    EventRow event = events.get(i);
                    long eventMs = event.eventTime.toEpochMilli();
                    long from = history.getBarStart(PERIOD, eventMs - WINDOW_BEFORE_MS);
                    long to = history.getBarStart(PERIOD, eventMs + WINDOW_AFTER_MS);

                    for (OfferSide side : new OfferSide[]{OfferSide.BID, OfferSide.ASK}) {
                        LoadResult result = loadBarsWithRetry(history, event, side, from, to);
                        writeStatus(statusWriter, event, side, result);

                        if (result.bars.isEmpty()) {
                            missingSideCount++;
                            System.err.printf(
                                    Locale.ROOT,
                                    "Event %d/%d %s %s MISSING after %d attempts%n",
                                    i + 1,
                                    events.size(),
                                    event.eventId,
                                    side,
                                    result.attempts
                            );
                            continue;
                        }

                        for (IBar bar : result.bars) {
                            writeBar(writer, event, side, bar);
                            rowCount++;
                        }
                        System.out.printf(
                                Locale.ROOT,
                                "Event %d/%d %s %s bars=%d attempts=%d%n",
                                i + 1,
                                events.size(),
                                event.eventId,
                                side,
                                result.bars.size(),
                                result.attempts
                        );
                    }
                }
            }

            moveIntoPlace(tempFile, finalFile);
            moveIntoPlace(tempStatusFile, finalStatusFile);
            System.out.println("Rows written: " + rowCount);
            System.out.println("Missing event/sides: " + missingSideCount);
            System.out.println("Saved: " + finalFile.toAbsolutePath());
            System.out.println("Status: " + finalStatusFile.toAbsolutePath());
        }

        private LoadResult loadBarsWithRetry(
                IHistory history,
                EventRow event,
                OfferSide side,
                long from,
                long to
        ) throws InterruptedException {
            String lastError = "no bars returned";

            for (int attempt = 1; attempt <= HISTORY_MAX_ATTEMPTS; attempt++) {
                try {
                    List<IBar> bars = history.getBars(INSTRUMENT, PERIOD, side, Filter.NO_FILTER, from, to);
                    if (bars != null && !bars.isEmpty()) {
                        return new LoadResult(bars, attempt, "");
                    }
                    lastError = "no bars returned";
                } catch (JFException ex) {
                    lastError = ex.getClass().getSimpleName() + ": " + safeMessage(ex);
                } catch (RuntimeException ex) {
                    lastError = ex.getClass().getSimpleName() + ": " + safeMessage(ex);
                }

                if (attempt < HISTORY_MAX_ATTEMPTS) {
                    long delayMs = HISTORY_RETRY_BASE_MS * (1L << (attempt - 1));
                    System.err.printf(
                            Locale.ROOT,
                            "History load retry %d/%d for %s %s after %s; waiting %.1fs%n",
                            attempt + 1,
                            HISTORY_MAX_ATTEMPTS,
                            event.eventId,
                            side,
                            lastError,
                            delayMs / 1000.0
                    );
                    Thread.sleep(delayMs);
                }
            }

            return new LoadResult(Collections.<IBar>emptyList(), HISTORY_MAX_ATTEMPTS, lastError);
        }

        private static String safeMessage(Throwable throwable) {
            String message = throwable.getMessage();
            return message == null ? throwable.toString() : message.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ');
        }

        private static void writeStatus(
                BufferedWriter writer,
                EventRow event,
                OfferSide side,
                LoadResult result
        ) throws IOException {
            writer.write(event.eventId);
            writer.write('\t');
            writer.write(event.eventType);
            writer.write('\t');
            writer.write(event.eventTime.toString());
            writer.write('\t');
            writer.write(side.name());
            writer.write('\t');
            writer.write(Integer.toString(result.attempts));
            writer.write('\t');
            writer.write(Integer.toString(result.bars.size()));
            writer.write('\t');
            writer.write(result.bars.isEmpty() ? "missing" : "ok");
            writer.write('\t');
            writer.write(result.error == null ? "" : result.error.replace('\t', ' ').replace('\r', ' ').replace('\n', ' '));
            writer.newLine();
        }

        private static void writeBar(BufferedWriter writer, EventRow event, OfferSide side, IBar bar)
                throws IOException {
            writer.write(event.eventId);
            writer.write('\t');
            writer.write(event.eventType);
            writer.write('\t');
            writer.write(event.referencePeriod);
            writer.write('\t');
            writer.write(event.eventTime.toString());
            writer.write('\t');
            writer.write(side.name());
            writer.write('\t');
            writer.write(Instant.ofEpochMilli(bar.getTime()).toString());
            writer.write('\t');
            writer.write(Double.toString(bar.getOpen()));
            writer.write('\t');
            writer.write(Double.toString(bar.getHigh()));
            writer.write('\t');
            writer.write(Double.toString(bar.getLow()));
            writer.write('\t');
            writer.write(Double.toString(bar.getClose()));
            writer.write('\t');
            writer.write(Double.toString(bar.getVolume()));
            writer.newLine();
        }

        private static void moveIntoPlace(Path tempFile, Path finalFile) throws IOException {
            try {
                Files.move(
                        tempFile,
                        finalFile,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE
                );
            } catch (AtomicMoveNotSupportedException ex) {
                Files.move(tempFile, finalFile, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        @Override
        public void onStop() {
            done.countDown();
        }

        @Override
        public void onTick(Instrument instrument, ITick tick) {
            // Read-only historical collector: live ticks are intentionally ignored.
        }

        @Override
        public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) {
            // Read-only historical collector: live bars are intentionally ignored.
        }

        @Override
        public void onMessage(IMessage message) {
            // No trading messages are consumed because this collector never places orders.
        }

        @Override
        public void onAccount(IAccount account) {
            // Account state is irrelevant to read-only historical collection.
        }
    }
}
