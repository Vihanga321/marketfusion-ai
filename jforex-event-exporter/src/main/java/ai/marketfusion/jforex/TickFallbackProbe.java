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
 * One-event, read-only diagnostic probe for MarketFusion AI.
 *
 * It downloads Dukascopy historical ticks, rebuilds BID and ASK M1 OHLC bars,
 * and validates the rebuilt BID bars against Dukascopy native BID M1 bars.
 * It never obtains IEngine and contains no order/position code.
 */
public final class TickFallbackProbe {
    private static final String DEMO_JNLP = "http://platform.dukascopy.com/demo_3/jforex_3.jnlp";
    private static final String USER_ENV = "DUKASCOPY_USER";
    private static final String PASSWORD_ENV = "DUKASCOPY_PASSWORD";
    private static final Instrument INSTRUMENT = Instrument.EURUSD;
    private static final Period PERIOD = Period.ONE_MIN;
    private static final long MINUTE_MS = 60_000L;
    private static final long WINDOW_BEFORE_MS = 10L * MINUTE_MS;
    private static final long WINDOW_AFTER_MS = 250L * MINUTE_MS;
    private static final long CONNECT_TIMEOUT_SECONDS = 45L;
    private static final double PRICE_TOLERANCE = 1.0e-8;

    private TickFallbackProbe() {
    }

    public static void main(String[] args) throws Exception {
        String username = requireEnvironment(USER_ENV);
        String password = requireEnvironment(PASSWORD_ENV);

        Path input = args.length >= 1
                ? Paths.get(args[0]).toAbsolutePath().normalize()
                : Paths.get("..", "data", "dukascopy", "events_for_jforex.tsv").toAbsolutePath().normalize();
        Path outputDirectory = args.length >= 2
                ? Paths.get(args[1]).toAbsolutePath().normalize()
                : Paths.get("..", "data", "dukascopy", "tick_probe").toAbsolutePath().normalize();

        List<EventRow> events = readEvents(input);
        if (events.isEmpty()) {
            throw new IllegalStateException("No events found in " + input);
        }
        EventRow event = events.get(0);

        System.out.println("MarketFusion JForex tick fallback probe");
        System.out.println("Event: " + event.eventId + " @ " + event.eventTime);
        System.out.println("Instrument: EUR/USD");
        System.out.println("Mode: read-only; historical ticks + native BID M1 validation");

        final IClient client = ClientFactory.getDefaultInstance();
        final CountDownLatch connected = new CountDownLatch(1);
        client.setSystemListener(new ISystemListener() {
            @Override
            public void onStart(long processId) {
                System.out.println("Probe strategy started: " + processId);
            }

            @Override
            public void onStop(long processId) {
                System.out.println("Probe strategy stopped: " + processId);
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
            ProbeStrategy strategy = new ProbeStrategy(event, outputDirectory);
            long strategyId = client.startStrategy(strategy);

            if (!strategy.await(10, TimeUnit.MINUTES)) {
                client.stopStrategy(strategyId);
                throw new IllegalStateException("Timed out waiting for tick fallback probe");
            }
            if (strategy.getFailure() != null) {
                throw new RuntimeException("Tick fallback probe failed", strategy.getFailure());
            }

            System.out.println("TICK_PROBE_STATUS: " + strategy.getStatus());
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

    private static final class ProbeStrategy implements IStrategy {
        private final EventRow event;
        private final Path outputDirectory;
        private final CountDownLatch done = new CountDownLatch(1);
        private volatile Throwable failure;
        private volatile String status = "FAIL";

        private ProbeStrategy(EventRow event, Path outputDirectory) {
            this.event = event;
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
                runProbe(context.getHistory());
            } catch (Throwable ex) {
                failure = ex;
                if (ex instanceof JFException) {
                    throw (JFException) ex;
                }
                throw new JFException("Tick fallback probe failed", ex);
            } finally {
                context.stop();
            }
        }

        private void runProbe(IHistory history) throws Exception {
            Files.createDirectories(outputDirectory);

            long eventMs = event.eventTime.toEpochMilli();
            long from = history.getBarStart(PERIOD, eventMs - WINDOW_BEFORE_MS);
            long lastBarStart = history.getBarStart(PERIOD, eventMs + WINDOW_AFTER_MS);
            long tickTo = lastBarStart + MINUTE_MS - 1L;

            List<IBar> nativeBid = history.getBars(
                    INSTRUMENT,
                    PERIOD,
                    OfferSide.BID,
                    Filter.NO_FILTER,
                    from,
                    lastBarStart
            );
            if (nativeBid == null || nativeBid.isEmpty()) {
                throw new JFException("Native BID M1 bars are unavailable; cannot validate tick reconstruction");
            }

            System.out.println("Native BID M1 bars: " + nativeBid.size());
            System.out.println("Requesting historical ticks: " + Instant.ofEpochMilli(from)
                    + " -> " + Instant.ofEpochMilli(tickTo));

            List<ITick> ticks = history.getTicks(INSTRUMENT, from, tickTo);
            if (ticks == null || ticks.isEmpty()) {
                throw new JFException("No historical ticks returned for probe window");
            }
            System.out.println("Historical ticks returned: " + ticks.size());

            TreeMap<Long, MinuteBar> rebuilt = new TreeMap<Long, MinuteBar>();
            for (ITick tick : ticks) {
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

            boolean pass = overlap > 0
                    && mismatched == 0
                    && missingRebuilt == 0
                    && overlap == nativeBid.size();
            status = pass ? "PASS" : "FAIL";

            Path barsFile = outputDirectory.resolve("tick_rebuilt_event_window.tsv");
            try (BufferedWriter writer = Files.newBufferedWriter(barsFile, StandardCharsets.UTF_8)) {
                writer.write("event_id\tevent_type\treference_period\tevent_timestamp_utc\tbar_time_utc\ttick_count"
                        + "\tbid_open\tbid_high\tbid_low\tbid_close"
                        + "\task_open\task_high\task_low\task_close"
                        + "\tbid_volume_sum\task_volume_sum\tspread_open\tspread_close");
                writer.newLine();
                for (MinuteBar bar : rebuilt.values()) {
                    writer.write(event.eventId);
                    writer.write('\t');
                    writer.write(event.eventType);
                    writer.write('\t');
                    writer.write(event.referencePeriod);
                    writer.write('\t');
                    writer.write(event.eventTime.toString());
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
            }

            Path validationFile = outputDirectory.resolve("tick_rebuild_validation.tsv");
            try (BufferedWriter writer = Files.newBufferedWriter(validationFile, StandardCharsets.UTF_8)) {
                writer.write("event_id\tnative_bid_bars\thistorical_ticks\trebuilt_minutes\toverlap_minutes\tmatched_minutes"
                        + "\tmismatched_minutes\tmissing_rebuilt_minutes\tprice_tolerance\tmax_abs_ohlc_diff\tstatus");
                writer.newLine();
                writer.write(event.eventId);
                writer.write('\t');
                writer.write(Integer.toString(nativeBid.size()));
                writer.write('\t');
                writer.write(Integer.toString(ticks.size()));
                writer.write('\t');
                writer.write(Integer.toString(rebuilt.size()));
                writer.write('\t');
                writer.write(Integer.toString(overlap));
                writer.write('\t');
                writer.write(Integer.toString(matched));
                writer.write('\t');
                writer.write(Integer.toString(mismatched));
                writer.write('\t');
                writer.write(Integer.toString(missingRebuilt));
                writer.write('\t');
                writer.write(Double.toString(PRICE_TOLERANCE));
                writer.write('\t');
                writer.write(Double.toString(maxAbsDiff));
                writer.write('\t');
                writer.write(status);
                writer.newLine();
            }

            System.out.println("Rebuilt M1 minutes: " + rebuilt.size());
            System.out.println("BID validation overlap: " + overlap + "/" + nativeBid.size());
            System.out.println("BID validation matched: " + matched);
            System.out.println("BID validation mismatched: " + mismatched);
            System.out.println("BID validation missing rebuilt: " + missingRebuilt);
            System.out.printf(Locale.ROOT, "BID max abs OHLC diff: %.10f%n", maxAbsDiff);
            System.out.println("Tick reconstruction validation: " + status);
            System.out.println("Bars: " + barsFile.toAbsolutePath());
            System.out.println("Validation: " + validationFile.toAbsolutePath());
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
            // Live ticks ignored. This probe uses historical ticks only.
        }

        @Override
        public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) {
            // Live bars ignored.
        }

        @Override
        public void onMessage(IMessage message) {
            // No trading messages are consumed.
        }

        @Override
        public void onAccount(IAccount account) {
            // Account state is irrelevant to this read-only probe.
        }
    }
}
