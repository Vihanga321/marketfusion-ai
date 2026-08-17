package ai.marketfusion.jforex;

import com.dukascopy.api.IAccount;
import com.dukascopy.api.IBar;
import com.dukascopy.api.IContext;
import com.dukascopy.api.IHistory;
import com.dukascopy.api.IMessage;
import com.dukascopy.api.IStrategy;
import com.dukascopy.api.ITick;
import com.dukascopy.api.Instrument;
import com.dukascopy.api.JFException;
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
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Resumable, read-only tick collector for the frozen 242-event V0.4A universe. */
public final class EventReactionCollector {
    static final String REACTION_CONTRACT = "v0.4a-event-reaction-v1";
    static final String VALIDATOR_CONTRACT = "v0.4a-native-bid-tick-rebuild-v3";
    static final String SOURCE = "DUKASCOPY_JFOREX_HISTORICAL_TICKS";
    static final int EXPECTED_EVENTS = 242;
    static final int EXPECTED_MINUTES = 261;
    private static final Instrument INSTRUMENT = Instrument.EURUSD;
    private static final String DEMO_JNLP = "http://platform.dukascopy.com/demo_3/jforex_3.jnlp";
    private static final long MINUTE_MS = 60_000L;
    private static final long HOUR_MS = 60L * MINUTE_MS;
    private static final int ATTEMPTS = 3;

    private EventReactionCollector() {
    }

    public static void main(String[] args) {
        int exit = 1;
        try {
            run(args);
            exit = 0;
        } catch (Throwable error) {
            System.err.println("EVENT_REACTION_COLLECTOR_FATAL: " + safe(error.getMessage()));
            error.printStackTrace(System.err);
        } finally {
            System.out.flush();
            System.err.flush();
            System.exit(exit);
        }
    }

    private static void run(String[] args) throws Exception {
        if (args.length != 3) {
            throw new IllegalArgumentException("Usage: EventReactionCollector <eligible.tsv> <output-directory> <repository-root>");
        }
        String username = requireEnvironment("DUKASCOPY_USER");
        String password = requireEnvironment("DUKASCOPY_PASSWORD");
        Path input = Paths.get(args[0]).toAbsolutePath().normalize();
        Path output = Paths.get(args[1]).toAbsolutePath().normalize();
        Path repositoryRoot = Paths.get(args[2]).toAbsolutePath().normalize();
        List<EventRow> events = readEligible(input);

        final IClient client = ClientFactory.getDefaultInstance();
        final CountDownLatch connected = new CountDownLatch(1);
        final CountDownLatch disconnected = new CountDownLatch(1);
        client.setSystemListener(new ISystemListener() {
            @Override public void onStart(long processId) { }
            @Override public void onStop(long processId) { }
            @Override public void onConnect() { connected.countDown(); }
            @Override public void onDisconnect() { disconnected.countDown(); }
        });
        CollectorStrategy strategy = null;
        try {
            client.connect(DEMO_JNLP, username, password);
            if (!connected.await(45L, TimeUnit.SECONDS) || !client.isConnected()) {
                throw new IllegalStateException("Dukascopy connection timeout");
            }
            client.setSubscribedInstruments(Collections.singleton(INSTRUMENT));
            strategy = new CollectorStrategy(events, output, repositoryRoot);
            long strategyId = client.startStrategy(strategy);
            if (!strategy.await(8L, TimeUnit.HOURS)) {
                client.stopStrategy(strategyId);
                throw new IllegalStateException("Reaction collection exceeded eight hours");
            }
            if (strategy.failure != null) {
                throw new RuntimeException("Reaction collection failed", strategy.failure);
            }
            System.out.println("REACTION_COLLECTION_STATUS: " + strategy.status);
        } finally {
            if (client.isConnected()) {
                client.disconnect();
                disconnected.await(8L, TimeUnit.SECONDS);
            }
        }
    }

    static final class EventRow {
        final String id;
        final String eventType;
        final String referencePeriod;
        final Instant eventTime;

        EventRow(String id, String eventType, String referencePeriod, Instant eventTime) {
            this.id = id;
            this.eventType = eventType;
            this.referencePeriod = referencePeriod;
            this.eventTime = eventTime;
        }
    }

    static List<EventRow> readEligible(Path input) throws IOException {
        if (!Files.isRegularFile(input)) throw new IOException("Missing eligible event TSV: " + input);
        List<EventRow> result = new ArrayList<EventRow>();
        Set<String> ids = new HashSet<String>();
        try (BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) throw new IOException("Eligible event TSV is empty");
            String[] names = header.split("\\t", -1);
            Map<String, Integer> positions = new HashMap<String, Integer>();
            for (int index = 0; index < names.length; index++) positions.put(names[index], index);
            for (String required : new String[]{
                    "event_id", "event_type", "reference_period", "event_timestamp_utc",
                    "production_status", "adjudication_class", "model_eligible_market_reaction",
                    "validator_version", "reaction_contract_version"
            }) {
                if (!positions.containsKey(required)) throw new IOException("Eligible TSV lacks " + required);
            }
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] values = line.split("\\t", -1);
                String id = value(values, positions, "event_id");
                if (!ids.add(id)) throw new IOException("Duplicate eligible event ID: " + id);
                if (!"PASS".equals(value(values, positions, "production_status"))
                        || !"STRICT_PASS".equals(value(values, positions, "adjudication_class"))
                        || !"True".equalsIgnoreCase(value(values, positions, "model_eligible_market_reaction"))
                        || !VALIDATOR_CONTRACT.equals(value(values, positions, "validator_version"))
                        || !REACTION_CONTRACT.equals(value(values, positions, "reaction_contract_version"))) {
                    throw new IOException("Non-eligible or stale-contract event in collector input: " + id);
                }
                Instant eventTime = Instant.parse(value(values, positions, "event_timestamp_utc"));
                if (eventTime.toEpochMilli() % MINUTE_MS != 0L) {
                    throw new IOException("Event timestamp is not an exact UTC minute: " + id);
                }
                result.add(new EventRow(
                        id, value(values, positions, "event_type"),
                        value(values, positions, "reference_period"), eventTime
                ));
            }
        }
        if (result.size() != EXPECTED_EVENTS) {
            throw new IOException("Collector input must contain exactly 242 strict-PASS events; found " + result.size());
        }
        return result;
    }

    private static String value(String[] values, Map<String, Integer> positions, String name) throws IOException {
        int index = positions.get(name);
        if (index >= values.length || values[index].trim().isEmpty()) throw new IOException("Empty " + name);
        return values[index].trim();
    }

    private static final class StatusRow {
        final EventRow event;
        final Instant from;
        final Instant to;
        final String retrievalStatus;
        final int rawTicks;
        final int acceptedTicks;
        final int minutes;
        final int invalidTicks;
        final int negativeSpreadTicks;
        final int duplicateTicks;
        final String windowFile;
        final String sha256;
        final boolean reused;
        final String error;

        StatusRow(EventRow event, Instant from, Instant to, String retrievalStatus, int rawTicks,
                  int acceptedTicks, int minutes, int invalidTicks, int negativeSpreadTicks,
                  int duplicateTicks, String windowFile, String sha256, boolean reused, String error) {
            this.event = event;
            this.from = from;
            this.to = to;
            this.retrievalStatus = retrievalStatus;
            this.rawTicks = rawTicks;
            this.acceptedTicks = acceptedTicks;
            this.minutes = minutes;
            this.invalidTicks = invalidTicks;
            this.negativeSpreadTicks = negativeSpreadTicks;
            this.duplicateTicks = duplicateTicks;
            this.windowFile = windowFile;
            this.sha256 = sha256;
            this.reused = reused;
            this.error = error;
        }
    }

    private static final class CollectorStrategy implements IStrategy {
        final List<EventRow> events;
        final Path output;
        final Path repositoryRoot;
        final CountDownLatch done = new CountDownLatch(1);
        final LinkedHashMap<String, StatusRow> statuses = new LinkedHashMap<String, StatusRow>();
        volatile Throwable failure;
        volatile String status = "ERROR";

        CollectorStrategy(List<EventRow> events, Path output, Path repositoryRoot) {
            this.events = events;
            this.output = output;
            this.repositoryRoot = repositoryRoot;
        }

        boolean await(long timeout, TimeUnit unit) throws InterruptedException {
            return done.await(timeout, unit);
        }

        @Override
        public void onStart(IContext context) throws JFException {
            try {
                Files.createDirectories(output.resolve("windows"));
                int index = 0;
                for (EventRow event : events) {
                    index++;
                    System.out.println("Reaction event " + index + "/" + events.size() + ": " + event.id);
                    StatusRow row;
                    try {
                        row = collectOrReuse(context.getHistory(), event);
                    } catch (Throwable error) {
                        long from = event.eventTime.toEpochMilli() - 10L * MINUTE_MS;
                        long to = event.eventTime.toEpochMilli() + 250L * MINUTE_MS;
                        row = new StatusRow(event, Instant.ofEpochMilli(from), Instant.ofEpochMilli(to),
                                "ERROR", 0, 0, 0, 0, 0, 0, "", "", false, safe(error.getMessage()));
                    }
                    statuses.put(event.id, row);
                    writeStatus();
                }
                int complete = 0;
                for (StatusRow row : statuses.values()) if ("COMPLETE".equals(row.retrievalStatus)) complete++;
                status = complete == events.size() ? "COMPLETE" : "INCOMPLETE";
            } catch (Throwable error) {
                failure = error;
            } finally {
                context.stop();
            }
        }

        private StatusRow collectOrReuse(IHistory history, EventRow event) throws Exception {
            long from = event.eventTime.toEpochMilli() - 10L * MINUTE_MS;
            long last = event.eventTime.toEpochMilli() + 250L * MINUTE_MS;
            long tickTo = last + MINUTE_MS - 1L;
            String stem = sha256Text(event.id).substring(0, 24);
            Path window = output.resolve("windows").resolve(stem + ".tsv");
            Path metadata = output.resolve("windows").resolve(stem + ".meta.tsv");
            StatusRow reused = validateCache(event, from, last, window, metadata);
            if (reused != null) return reused;

            List<TickBarReconstructor.Quote> quotes = new ArrayList<TickBarReconstructor.Quote>();
            long sequence = 0L;
            int emptyChunks = 0;
            for (long cursor = from; cursor <= tickTo; ) {
                long chunkTo = Math.min(tickTo, ((cursor / HOUR_MS) + 1L) * HOUR_MS - 1L);
                List<ITick> ticks = retryTicks(history, cursor, chunkTo);
                if (ticks.isEmpty()) emptyChunks++;
                for (ITick tick : ticks) {
                    quotes.add(new TickBarReconstructor.Quote(
                            tick.getTime(), tick.getBid(), tick.getAsk(), sequence++
                    ));
                }
                cursor = chunkTo + 1L;
            }
            if (emptyChunks > 0) {
                return status(event, from, last, "TEMPORARILY_UNAVAILABLE", quotes.size(), null,
                        "provider returned zero ticks for " + emptyChunks + " hourly chunk(s)");
            }
            EventReactionReconstructor.Result rebuilt = EventReactionReconstructor.rebuild(quotes, from, last);
            if (rebuilt.invalidQuoteTicks > 0 || rebuilt.negativeSpreadTicks > 0) {
                return status(event, from, last, "DATA_QUALITY_FAILURE", quotes.size(), rebuilt,
                        "invalid or negative-spread ticks were rejected");
            }
            if (rebuilt.bars.size() != EXPECTED_MINUTES) {
                return status(event, from, last, "TEMPORARILY_UNAVAILABLE", quotes.size(), rebuilt,
                        "tick history reconstructed " + rebuilt.bars.size() + "/261 minutes; no fill applied");
            }
            writeWindow(window, event, rebuilt.bars);
            String hash = sha256File(window);
            String relative = relative(window);
            writeMetadata(metadata, event, from, last, relative, hash, rebuilt);
            return new StatusRow(event, Instant.ofEpochMilli(from), Instant.ofEpochMilli(last), "COMPLETE",
                    quotes.size(), rebuilt.acceptedTicks, rebuilt.bars.size(), rebuilt.invalidQuoteTicks,
                    rebuilt.negativeSpreadTicks, rebuilt.duplicateTicks, relative, hash, false, "");
        }

        private StatusRow status(EventRow event, long from, long last, String state, int raw,
                                 EventReactionReconstructor.Result rebuilt, String error) {
            return new StatusRow(event, Instant.ofEpochMilli(from), Instant.ofEpochMilli(last), state,
                    raw, rebuilt == null ? 0 : rebuilt.acceptedTicks,
                    rebuilt == null ? 0 : rebuilt.bars.size(),
                    rebuilt == null ? 0 : rebuilt.invalidQuoteTicks,
                    rebuilt == null ? 0 : rebuilt.negativeSpreadTicks,
                    rebuilt == null ? 0 : rebuilt.duplicateTicks, "", "", false, error);
        }

        private List<ITick> retryTicks(IHistory history, long from, long to) throws InterruptedException {
            for (int attempt = 1; attempt <= ATTEMPTS; attempt++) {
                try {
                    List<ITick> ticks = history.getTicks(INSTRUMENT, from, to);
                    if (ticks != null && !ticks.isEmpty()) return ticks;
                } catch (JFException error) {
                    System.err.println("Tick history attempt " + attempt + " failed: " + safe(error.getMessage()));
                }
                if (attempt < ATTEMPTS) Thread.sleep(2_000L * (1L << (attempt - 1)));
            }
            return Collections.emptyList();
        }

        private StatusRow validateCache(EventRow event, long from, long last, Path window, Path metadata) {
            try {
                if (!Files.isRegularFile(window) || !Files.isRegularFile(metadata)) return null;
                Map<String, String> values = readTwoRowTsv(metadata);
                String relative = relative(window);
                if (!event.id.equals(values.get("event_id"))
                        || !event.eventTime.toString().equals(values.get("event_timestamp_utc"))
                        || !Instant.ofEpochMilli(from).toString().equals(values.get("window_start_utc"))
                        || !Instant.ofEpochMilli(last).toString().equals(values.get("window_end_utc"))
                        || !REACTION_CONTRACT.equals(values.get("reaction_contract_version"))
                        || !SOURCE.equals(values.get("source"))
                        || !"COMPLETE".equals(values.get("retrieval_status"))
                        || !relative.equals(values.get("window_file"))
                        || !Integer.toString(EXPECTED_MINUTES).equals(values.get("minute_count"))) return null;
                String actualHash = sha256File(window);
                if (!actualHash.equals(values.get("data_sha256"))) return null;
                validateWindowFile(window, event, from, last);
                return new StatusRow(event, Instant.ofEpochMilli(from), Instant.ofEpochMilli(last), "COMPLETE",
                        integer(values, "raw_tick_count"), integer(values, "accepted_tick_count"),
                        EXPECTED_MINUTES, integer(values, "invalid_tick_count"),
                        integer(values, "negative_spread_tick_count"), integer(values, "duplicate_tick_count"),
                        relative, actualHash, true, "");
            } catch (Exception ignored) {
                return null;
            }
        }

        private void validateWindowFile(Path window, EventRow event, long from, long last) throws IOException {
            try (BufferedReader reader = Files.newBufferedReader(window, StandardCharsets.UTF_8)) {
                String header = reader.readLine();
                if (header == null || !header.startsWith("event_id\tevent_timestamp_utc\tminute_utc\t")) {
                    throw new IOException("invalid cached window header");
                }
                String line;
                int rows = 0;
                long expected = from;
                while ((line = reader.readLine()) != null) {
                    String[] values = line.split("\\t", -1);
                    if (values.length != 20 || !event.id.equals(values[0])
                            || !event.eventTime.toString().equals(values[1])
                            || !Instant.ofEpochMilli(expected).toString().equals(values[2])) {
                        throw new IOException("cached window identity or minute mismatch");
                    }
                    double[] market = new double[16];
                    for (int index = 3; index < 19; index++) {
                        double value = Double.parseDouble(values[index]);
                        if (Double.isNaN(value) || Double.isInfinite(value) || value < 0.0) {
                            throw new IOException("cached window has invalid market values");
                        }
                        market[index - 3] = value;
                    }
                    for (int offset : new int[]{0, 4, 8}) {
                        double open = market[offset];
                        double high = market[offset + 1];
                        double low = market[offset + 2];
                        double close = market[offset + 3];
                        if (open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0
                                || high < Math.max(Math.max(open, close), low)
                                || low > Math.min(Math.min(open, close), high)) {
                            throw new IOException("cached BID/ASK/MID OHLC is invalid");
                        }
                    }
                    if (market[4] < market[0] || market[7] < market[3]
                            || market[13] < Math.max(Math.max(market[12], market[15]), market[14])
                            || market[14] > Math.min(Math.min(market[12], market[15]), market[13])) {
                        throw new IOException("cached spread geometry is invalid");
                    }
                    if (Math.abs(market[8] - (market[0] + market[4]) / 2.0) > 5.0e-10
                            || Math.abs(market[11] - (market[3] + market[7]) / 2.0) > 5.0e-10
                            || Math.abs(market[12] - (market[4] - market[0])) > 5.0e-10
                            || Math.abs(market[15] - (market[7] - market[3])) > 5.0e-10) {
                        throw new IOException("cached MID/spread open or close is inconsistent");
                    }
                    if (Integer.parseInt(values[19]) <= 0) throw new IOException("cached minute has no ticks");
                    rows++;
                    expected += MINUTE_MS;
                }
                if (rows != EXPECTED_MINUTES || expected - MINUTE_MS != last) {
                    throw new IOException("cached window minute coverage mismatch");
                }
            }
        }

        private void writeWindow(Path destination, EventRow event,
                                 TreeMap<Long, EventReactionReconstructor.MinuteBar> bars) throws IOException {
            Path temporary = temporary(destination);
            try (BufferedWriter writer = Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) {
                writer.write("event_id\tevent_timestamp_utc\tminute_utc\t"
                        + "bid_open\tbid_high\tbid_low\tbid_close\t"
                        + "ask_open\task_high\task_low\task_close\t"
                        + "mid_open\tmid_high\tmid_low\tmid_close\t"
                        + "spread_open\tspread_high\tspread_low\tspread_close\ttick_count");
                writer.newLine();
                for (Map.Entry<Long, EventReactionReconstructor.MinuteBar> entry : bars.entrySet()) {
                    EventReactionReconstructor.MinuteBar bar = entry.getValue();
                    writer.write(join(new String[]{
                            event.id, event.eventTime.toString(), Instant.ofEpochMilli(entry.getKey()).toString(),
                            number(bar.bidOpen), number(bar.bidHigh), number(bar.bidLow), number(bar.bidClose),
                            number(bar.askOpen), number(bar.askHigh), number(bar.askLow), number(bar.askClose),
                            number(bar.midOpen), number(bar.midHigh), number(bar.midLow), number(bar.midClose),
                            number(bar.spreadOpen), number(bar.spreadHigh), number(bar.spreadLow), number(bar.spreadClose),
                            Integer.toString(bar.tickCount)
                    }));
                    writer.newLine();
                }
            }
            publish(temporary, destination);
        }

        private void writeMetadata(Path destination, EventRow event, long from, long last, String windowFile,
                                   String hash, EventReactionReconstructor.Result rebuilt) throws IOException {
            Path temporary = temporary(destination);
            try (BufferedWriter writer = Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) {
                writer.write("event_id\tevent_timestamp_utc\twindow_start_utc\twindow_end_utc\t"
                        + "reaction_contract_version\tsource\tretrieval_status\traw_tick_count\t"
                        + "accepted_tick_count\tminute_count\tinvalid_tick_count\tnegative_spread_tick_count\t"
                        + "duplicate_tick_count\twindow_file\tdata_sha256");
                writer.newLine();
                writer.write(join(new String[]{
                        event.id, event.eventTime.toString(), Instant.ofEpochMilli(from).toString(),
                        Instant.ofEpochMilli(last).toString(), REACTION_CONTRACT, SOURCE, "COMPLETE",
                        Integer.toString(rebuilt.acceptedTicks + rebuilt.duplicateTicks + rebuilt.invalidQuoteTicks),
                        Integer.toString(rebuilt.acceptedTicks), Integer.toString(rebuilt.bars.size()),
                        Integer.toString(rebuilt.invalidQuoteTicks), Integer.toString(rebuilt.negativeSpreadTicks),
                        Integer.toString(rebuilt.duplicateTicks), windowFile, hash
                }));
                writer.newLine();
            }
            publish(temporary, destination);
        }

        private void writeStatus() throws IOException {
            Path destination = output.resolve("reaction_extraction_status.tsv");
            Path temporary = temporary(destination);
            try (BufferedWriter writer = Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) {
                writer.write("event_id\tevent_type\treference_period\tevent_timestamp_utc\t"
                        + "reaction_contract_version\tsource\twindow_start_utc\twindow_end_utc\t"
                        + "retrieval_status\traw_tick_count\taccepted_tick_count\tminute_count\t"
                        + "invalid_tick_count\tnegative_spread_tick_count\tduplicate_tick_count\t"
                        + "window_file\tdata_sha256\treused_cache\terror");
                writer.newLine();
                for (StatusRow row : statuses.values()) {
                    writer.write(join(new String[]{
                            row.event.id, row.event.eventType, row.event.referencePeriod, row.event.eventTime.toString(),
                            REACTION_CONTRACT, SOURCE, row.from.toString(), row.to.toString(), row.retrievalStatus,
                            Integer.toString(row.rawTicks), Integer.toString(row.acceptedTicks),
                            Integer.toString(row.minutes), Integer.toString(row.invalidTicks),
                            Integer.toString(row.negativeSpreadTicks), Integer.toString(row.duplicateTicks),
                            row.windowFile, row.sha256, Boolean.toString(row.reused), row.error
                    }));
                    writer.newLine();
                }
            }
            publish(temporary, destination);
        }

        private String relative(Path path) {
            return repositoryRoot.relativize(path.toAbsolutePath().normalize()).toString().replace('\\', '/');
        }

        @Override public void onStop() { done.countDown(); }
        @Override public void onTick(Instrument instrument, ITick tick) { }
        @Override public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) { }
        @Override public void onMessage(IMessage message) { }
        @Override public void onAccount(IAccount account) { }
    }

    private static Map<String, String> readTwoRowTsv(Path path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            String values = reader.readLine();
            if (header == null || values == null || reader.readLine() != null) throw new IOException("invalid metadata TSV");
            String[] names = header.split("\\t", -1);
            String[] data = values.split("\\t", -1);
            if (names.length != data.length) throw new IOException("metadata width mismatch");
            Map<String, String> result = new HashMap<String, String>();
            for (int index = 0; index < names.length; index++) result.put(names[index], data[index]);
            return result;
        }
    }

    private static int integer(Map<String, String> values, String key) {
        return Integer.parseInt(values.get(key));
    }

    private static String requireEnvironment(String name) {
        String value = System.getenv(name);
        if (value == null || value.trim().isEmpty()) throw new IllegalStateException("Missing required environment variable: " + name);
        return value;
    }

    private static String number(double value) {
        return String.format(Locale.ROOT, "%.10f", value);
    }

    private static String join(String[] values) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) result.append('\t');
            result.append(safe(values[index]));
        }
        return result.toString();
    }

    private static String safe(String value) {
        return value == null ? "" : value.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ');
    }

    private static Path temporary(Path destination) throws IOException {
        Files.createDirectories(destination.toAbsolutePath().normalize().getParent());
        return destination.resolveSibling("." + destination.getFileName() + ".tmp");
    }

    private static void publish(Path source, Path destination) throws IOException {
        try {
            Files.move(source, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException error) {
            Files.move(source, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static String sha256Text(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return hex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException(error);
        }
    }

    private static String sha256File(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = new byte[1024 * 1024];
            try (java.io.InputStream input = Files.newInputStream(path)) {
                int read;
                while ((read = input.read(bytes)) >= 0) if (read > 0) digest.update(bytes, 0, read);
            }
            return hex(digest.digest());
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException(error);
        }
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder();
        for (byte value : bytes) result.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        return result.toString();
    }
}
