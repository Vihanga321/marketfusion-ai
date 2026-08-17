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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Read-only, four-event-only Dukascopy mismatch diagnostic. */
public final class TickMismatchDiagnostic {
    private static final String DEMO_JNLP = "http://platform.dukascopy.com/demo_3/jforex_3.jnlp";
    private static final Instrument INSTRUMENT = Instrument.EURUSD;
    private static final Period PERIOD = Period.ONE_MIN;
    private static final long MINUTE_MS = 60_000L;
    private static final long HOUR_MS = 60L * MINUTE_MS;
    private static final int EXPECTED_MINUTES = 261;
    private static final int ATTEMPTS = 3;
    private static final Set<String> APPROVED_IDS = Collections.unmodifiableSet(new HashSet<String>(Arrays.asList(
            "bls:us_employment_situation:20220902T1230Z",
            "bls:us_cpi_release:20230913T1230Z",
            "bls:us_employment_situation:20250404T1230Z",
            "bls:us_cpi_release:20260512T1230Z"
    )));

    private TickMismatchDiagnostic() {
    }

    public static void main(String[] args) {
        int exit = 1;
        try {
            run(args);
            exit = 0;
        } catch (Throwable error) {
            System.err.println("MISMATCH_DIAGNOSTIC_FATAL: " + safe(error.getMessage()));
            error.printStackTrace(System.err);
        } finally {
            System.out.flush();
            System.err.flush();
            System.exit(exit);
        }
    }

    private static void run(String[] args) throws Exception {
        if (args.length != 4) {
            throw new IllegalArgumentException(
                    "Usage: TickMismatchDiagnostic <four-events.tsv> <minutes.tsv> <tick-context.tsv> <report.txt>"
            );
        }
        String username = requireEnvironment("DUKASCOPY_USER");
        String password = requireEnvironment("DUKASCOPY_PASSWORD");
        List<EventRow> events = readApprovedEvents(Paths.get(args[0]));
        final IClient client = ClientFactory.getDefaultInstance();
        final CountDownLatch connected = new CountDownLatch(1);
        final CountDownLatch disconnected = new CountDownLatch(1);
        client.setSystemListener(new ISystemListener() {
            @Override public void onStart(long processId) { }
            @Override public void onStop(long processId) { }
            @Override public void onConnect() { connected.countDown(); }
            @Override public void onDisconnect() { disconnected.countDown(); }
        });

        DiagnosticStrategy strategy = null;
        try {
            client.connect(DEMO_JNLP, username, password);
            if (!connected.await(45L, TimeUnit.SECONDS) || !client.isConnected()) {
                throw new IllegalStateException("Dukascopy connection timeout");
            }
            client.setSubscribedInstruments(Collections.singleton(INSTRUMENT));
            strategy = new DiagnosticStrategy(
                    events, Paths.get(args[1]), Paths.get(args[2]), Paths.get(args[3])
            );
            long id = client.startStrategy(strategy);
            if (!strategy.await(30L, TimeUnit.MINUTES)) {
                client.stopStrategy(id);
                throw new IllegalStateException("Targeted diagnostic timed out");
            }
            if (strategy.failure != null) {
                throw new RuntimeException("Targeted diagnostic failed", strategy.failure);
            }
            System.out.println("TARGETED_DUKASCOPY_DIAGNOSTIC_STATUS: " + strategy.status);
        } finally {
            if (client.isConnected()) {
                client.disconnect();
                disconnected.await(8L, TimeUnit.SECONDS);
            }
        }
    }

    private static final class EventRow {
        final String id;
        final Instant time;

        EventRow(String id, Instant time) {
            this.id = id;
            this.time = time;
        }
    }

    private static List<EventRow> readApprovedEvents(Path input) throws IOException {
        List<EventRow> result = new ArrayList<EventRow>();
        Set<String> ids = new HashSet<String>();
        try (BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) throw new IOException("Empty diagnostic event TSV");
            String[] names = header.split("\\t", -1);
            Map<String, Integer> positions = new HashMap<String, Integer>();
            for (int index = 0; index < names.length; index++) positions.put(names[index], index);
            if (!positions.containsKey("event_id") || !positions.containsKey("event_timestamp_utc")) {
                throw new IOException("Diagnostic TSV requires event_id and event_timestamp_utc");
            }
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] values = line.split("\\t", -1);
                String id = values[positions.get("event_id")].trim();
                if (!APPROVED_IDS.contains(id) || !ids.add(id)) {
                    throw new IOException("Unapproved or duplicate diagnostic event_id: " + id);
                }
                Instant time = Instant.parse(values[positions.get("event_timestamp_utc")].trim());
                if (time.toEpochMilli() % MINUTE_MS != 0L) {
                    throw new IOException("Event is not minute aligned: " + id);
                }
                if (!time.equals(approvedTime(id))) {
                    throw new IOException("Diagnostic event timestamp does not match the approved ID: " + id);
                }
                result.add(new EventRow(id, time));
            }
        }
        if (!ids.equals(APPROVED_IDS) || result.size() != 4) {
            throw new IOException("Diagnostic input must contain exactly the four approved event IDs");
        }
        return result;
    }

    private static Instant approvedTime(String id) {
        if ("bls:us_employment_situation:20220902T1230Z".equals(id)) {
            return Instant.parse("2022-09-02T12:30:00Z");
        }
        if ("bls:us_cpi_release:20230913T1230Z".equals(id)) {
            return Instant.parse("2023-09-13T12:30:00Z");
        }
        if ("bls:us_employment_situation:20250404T1230Z".equals(id)) {
            return Instant.parse("2025-04-04T12:30:00Z");
        }
        if ("bls:us_cpi_release:20260512T1230Z".equals(id)) {
            return Instant.parse("2026-05-12T12:30:00Z");
        }
        throw new IllegalArgumentException("Unapproved event ID: " + id);
    }

    private static final class EventEvidence {
        final EventRow event;
        final int nativeCount;
        final int tickCount;
        final int emptyNativeChunks;
        final int emptyTickChunks;
        final List<MismatchDiagnosticAnalyzer.Issue> issues;

        EventEvidence(EventRow event, int nativeCount, int tickCount, int emptyNativeChunks,
                      int emptyTickChunks, List<MismatchDiagnosticAnalyzer.Issue> issues) {
            this.event = event;
            this.nativeCount = nativeCount;
            this.tickCount = tickCount;
            this.emptyNativeChunks = emptyNativeChunks;
            this.emptyTickChunks = emptyTickChunks;
            this.issues = issues;
        }
    }

    private static final class DiagnosticStrategy implements IStrategy {
        final List<EventRow> events;
        final Path minutesOutput;
        final Path contextOutput;
        final Path reportOutput;
        final CountDownLatch done = new CountDownLatch(1);
        volatile Throwable failure;
        volatile String status = "ERROR";

        DiagnosticStrategy(List<EventRow> events, Path minutesOutput, Path contextOutput, Path reportOutput) {
            this.events = events;
            this.minutesOutput = minutesOutput;
            this.contextOutput = contextOutput;
            this.reportOutput = reportOutput;
        }

        boolean await(long timeout, TimeUnit unit) throws InterruptedException {
            return done.await(timeout, unit);
        }

        @Override
        public void onStart(IContext context) throws JFException {
            try {
                List<EventEvidence> evidence = new ArrayList<EventEvidence>();
                for (EventRow event : events) {
                    System.out.println("Diagnosing only: " + event.id);
                    evidence.add(loadAndAnalyze(context.getHistory(), event));
                }
                writeOutputs(evidence);
                status = isComplete(evidence) ? "COMPLETE" : "INCOMPLETE";
            } catch (Throwable error) {
                failure = error;
            } finally {
                context.stop();
            }
        }

        private EventEvidence loadAndAnalyze(IHistory history, EventRow event) throws Exception {
            long from = history.getBarStart(PERIOD, event.time.toEpochMilli() - 10L * MINUTE_MS);
            long last = history.getBarStart(PERIOD, event.time.toEpochMilli() + 250L * MINUTE_MS);
            long tickTo = last + MINUTE_MS - 1L;
            TreeMap<Long, MismatchDiagnosticAnalyzer.NativeBidBar> nativeBars =
                    new TreeMap<Long, MismatchDiagnosticAnalyzer.NativeBidBar>();
            int emptyNativeChunks = 0;

            List<IBar> whole = retryBars(history, from, last);
            addBars(nativeBars, whole, from, last);
            if (nativeBars.size() != EXPECTED_MINUTES) {
                nativeBars.clear();
                for (long cursor = from; cursor <= last; ) {
                    long end = Math.min(last, ((cursor / HOUR_MS) + 1L) * HOUR_MS - MINUTE_MS);
                    List<IBar> chunk = retryBars(history, cursor, end);
                    if (chunk.isEmpty()) emptyNativeChunks++;
                    addBars(nativeBars, chunk, from, last);
                    cursor = end + MINUTE_MS;
                }
            }

            List<TickBarReconstructor.Quote> quotes = new ArrayList<TickBarReconstructor.Quote>();
            long sequence = 0L;
            int emptyTickChunks = 0;
            for (long cursor = from; cursor <= tickTo; ) {
                long end = Math.min(tickTo, ((cursor / HOUR_MS) + 1L) * HOUR_MS - 1L);
                List<ITick> ticks = retryTicks(history, cursor, end);
                if (ticks.isEmpty()) emptyTickChunks++;
                for (ITick tick : ticks) {
                    quotes.add(new TickBarReconstructor.Quote(
                            tick.getTime(), tick.getBid(), tick.getAsk(), sequence++
                    ));
                }
                cursor = end + 1L;
            }
            List<MismatchDiagnosticAnalyzer.Issue> issues = MismatchDiagnosticAnalyzer.analyze(
                    event.id, event.time, from, last, nativeBars, quotes
            );
            return new EventEvidence(
                    event, nativeBars.size(), quotes.size(), emptyNativeChunks, emptyTickChunks, issues
            );
        }

        private List<IBar> retryBars(IHistory history, long from, long to) throws InterruptedException {
            for (int attempt = 1; attempt <= ATTEMPTS; attempt++) {
                try {
                    List<IBar> bars = history.getBars(
                            INSTRUMENT, PERIOD, OfferSide.BID, Filter.NO_FILTER, from, to
                    );
                    if (bars != null && !bars.isEmpty()) return bars;
                } catch (JFException error) {
                    System.err.println("native BID attempt " + attempt + " failed: " + safe(error.getMessage()));
                }
                backoff(attempt);
            }
            return Collections.emptyList();
        }

        private List<ITick> retryTicks(IHistory history, long from, long to) throws InterruptedException {
            for (int attempt = 1; attempt <= ATTEMPTS; attempt++) {
                try {
                    List<ITick> ticks = history.getTicks(INSTRUMENT, from, to);
                    if (ticks != null && !ticks.isEmpty()) return ticks;
                } catch (JFException error) {
                    System.err.println("tick attempt " + attempt + " failed: " + safe(error.getMessage()));
                }
                backoff(attempt);
            }
            return Collections.emptyList();
        }

        private void backoff(int attempt) throws InterruptedException {
            if (attempt < ATTEMPTS) Thread.sleep(2_000L * (1L << (attempt - 1)));
        }

        private void addBars(TreeMap<Long, MismatchDiagnosticAnalyzer.NativeBidBar> target,
                             List<IBar> bars, long from, long last) {
            for (IBar bar : bars) {
                if (bar.getTime() >= from && bar.getTime() <= last) {
                    target.put(bar.getTime(), new MismatchDiagnosticAnalyzer.NativeBidBar(
                            bar.getOpen(), bar.getHigh(), bar.getLow(), bar.getClose()
                    ));
                }
            }
        }

        private void writeOutputs(List<EventEvidence> eventsEvidence) throws IOException {
            ensureParent(minutesOutput);
            ensureParent(contextOutput);
            ensureParent(reportOutput);
            Path minuteTemp = temp(minutesOutput);
            Path contextTemp = temp(contextOutput);
            Path reportTemp = temp(reportOutput);
            try {
                writeMinutes(minuteTemp, eventsEvidence);
                writeContext(contextTemp, eventsEvidence);
                writeReport(reportTemp, eventsEvidence);
                publish(minuteTemp, minutesOutput);
                publish(contextTemp, contextOutput);
                publish(reportTemp, reportOutput);
            } finally {
                Files.deleteIfExists(minuteTemp);
                Files.deleteIfExists(contextTemp);
                Files.deleteIfExists(reportTemp);
            }
        }

        private void writeMinutes(Path path, List<EventEvidence> eventsEvidence) throws IOException {
            try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                writer.write("event_id\tevent_timestamp_utc\tminute_utc\tcomparison_status\t"
                        + "native_open\tnative_high\tnative_low\tnative_close\t"
                        + "rebuilt_open\trebuilt_high\trebuilt_low\trebuilt_close\t"
                        + "open_diff\thigh_diff\tlow_diff\tclose_diff\tmax_diff\ttick_count\t"
                        + "raw_tick_count\tinvalid_tick_count\texact_duplicates_removed\t"
                        + "exact_duplicate_removal_changed_minute\texact_duplicate_removal_changed_ohlc\t"
                        + "classification\tevidence");
                writer.newLine();
                for (EventEvidence event : eventsEvidence) for (MismatchDiagnosticAnalyzer.Issue issue : event.issues) {
                    MismatchDiagnosticAnalyzer.NativeBidBar nativeBar = issue.nativeBar;
                    TickBarReconstructor.MinuteBar rebuilt = issue.rebuiltBar;
                    List<String> values = Arrays.asList(
                            issue.eventId, issue.eventTime.toString(), MismatchDiagnosticAnalyzer.timestamp(issue.minute),
                            issue.comparisonStatus,
                            nativeBar == null ? "" : MismatchDiagnosticAnalyzer.number(nativeBar.open),
                            nativeBar == null ? "" : MismatchDiagnosticAnalyzer.number(nativeBar.high),
                            nativeBar == null ? "" : MismatchDiagnosticAnalyzer.number(nativeBar.low),
                            nativeBar == null ? "" : MismatchDiagnosticAnalyzer.number(nativeBar.close),
                            rebuilt == null ? "" : MismatchDiagnosticAnalyzer.number(rebuilt.bidOpen),
                            rebuilt == null ? "" : MismatchDiagnosticAnalyzer.number(rebuilt.bidHigh),
                            rebuilt == null ? "" : MismatchDiagnosticAnalyzer.number(rebuilt.bidLow),
                            rebuilt == null ? "" : MismatchDiagnosticAnalyzer.number(rebuilt.bidClose),
                            MismatchDiagnosticAnalyzer.number(issue.openDiff),
                            MismatchDiagnosticAnalyzer.number(issue.highDiff),
                            MismatchDiagnosticAnalyzer.number(issue.lowDiff),
                            MismatchDiagnosticAnalyzer.number(issue.closeDiff),
                            MismatchDiagnosticAnalyzer.number(issue.maxDiff),
                            Integer.toString(issue.tickCount), Integer.toString(issue.rawTickCount),
                            Integer.toString(issue.invalidTickCount), Integer.toString(issue.duplicateTickCount),
                            Boolean.toString(issue.duplicateRemovalChangedMinute),
                            Boolean.toString(issue.duplicateRemovalChangedOhlc),
                            issue.classification.name(), issue.evidence
                    );
                    writer.write(join(values)); writer.newLine();
                }
            }
        }

        private void writeContext(Path path, List<EventEvidence> eventsEvidence) throws IOException {
            try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                writer.write("event_id\tevent_timestamp_utc\tminute_utc\tcontext_scope\tcontext_position\t"
                        + "context_rank\ttick_timestamp_utc\tbid\task\tprovider_sequence\taccepted\t"
                        + "rejection_reason\texact_duplicate\texact_duplicate_removal_changed_minute");
                writer.newLine();
                for (EventEvidence event : eventsEvidence) for (MismatchDiagnosticAnalyzer.Issue issue : event.issues) {
                    for (MismatchDiagnosticAnalyzer.ContextTick tick : issue.context) {
                        writer.write(join(Arrays.asList(
                                issue.eventId, issue.eventTime.toString(), MismatchDiagnosticAnalyzer.timestamp(issue.minute),
                                tick.scope, tick.position, Integer.toString(tick.rank),
                                MismatchDiagnosticAnalyzer.timestamp(tick.quote.time),
                                MismatchDiagnosticAnalyzer.number(tick.quote.bid),
                                MismatchDiagnosticAnalyzer.number(tick.quote.ask),
                                Long.toString(tick.quote.sequence), Boolean.toString(tick.accepted),
                                tick.rejectionReason, Boolean.toString(tick.exactDuplicate),
                                Boolean.toString(issue.duplicateRemovalChangedMinute)
                        ))); writer.newLine();
                    }
                }
            }
        }

        private void writeReport(Path path, List<EventEvidence> eventsEvidence) throws IOException {
            try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                writer.write("MARKETFUSION DUKASCOPY TARGETED MISMATCH DIAGNOSTIC"); writer.newLine();
                writer.write("Scope: exactly four approved production MISMATCH events; no other event was requested"); writer.newLine();
                writer.write("Mode: read-only historical BID M1 and tick retrieval; no trading API"); writer.newLine();
                writer.write("PRICE_TOLERANCE: 1.0e-8 (unchanged)"); writer.newLine();
                writer.write("Production validation outcome: NOT MODIFIED"); writer.newLine();
                writer.newLine();
                for (EventEvidence event : eventsEvidence) {
                    writer.write("EVENT: " + event.event.id); writer.newLine();
                    writer.write("event_timestamp_utc: " + event.event.time); writer.newLine();
                    writer.write("native_bid_minutes: " + event.nativeCount + "/" + EXPECTED_MINUTES); writer.newLine();
                    writer.write("historical_ticks: " + event.tickCount); writer.newLine();
                    writer.write("empty_native_chunks_after_retries: " + event.emptyNativeChunks); writer.newLine();
                    writer.write("empty_tick_chunks_after_retries: " + event.emptyTickChunks); writer.newLine();
                    writer.write("affected_minutes: " + event.issues.size()); writer.newLine();
                    for (MismatchDiagnosticAnalyzer.Issue issue : event.issues) {
                        writer.write("- " + MismatchDiagnosticAnalyzer.timestamp(issue.minute)
                                + " | " + issue.comparisonStatus + " | " + issue.classification.name()
                                + " | max_diff=" + MismatchDiagnosticAnalyzer.number(issue.maxDiff)
                                + " | " + issue.evidence);
                        writer.newLine();
                    }
                    writer.newLine();
                }
                writer.write("DIAGNOSTIC_STATUS: " + (isComplete(eventsEvidence) ? "COMPLETE" : "INCOMPLETE"));
                writer.newLine();
            }
        }

        private boolean isComplete(List<EventEvidence> values) {
            if (values.size() != 4) return false;
            for (EventEvidence value : values) {
                if (value.nativeCount != EXPECTED_MINUTES || value.emptyNativeChunks > 0
                        || value.emptyTickChunks > 0) return false;
            }
            return true;
        }

        @Override public void onStop() { done.countDown(); }
        @Override public void onTick(Instrument instrument, ITick tick) { }
        @Override public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) { }
        @Override public void onMessage(IMessage message) { }
        @Override public void onAccount(IAccount account) { }
    }

    private static String requireEnvironment(String name) {
        String value = System.getenv(name);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalStateException("Missing required environment variable: " + name);
        }
        return value;
    }

    private static void ensureParent(Path path) throws IOException {
        Path parent = path.toAbsolutePath().normalize().getParent();
        if (parent != null) Files.createDirectories(parent);
    }

    private static Path temp(Path destination) {
        Path absolute = destination.toAbsolutePath().normalize();
        return absolute.resolveSibling("." + absolute.getFileName() + ".tmp");
    }

    private static void publish(Path source, Path destination) throws IOException {
        try {
            Files.move(source, destination.toAbsolutePath().normalize(),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException error) {
            Files.move(source, destination.toAbsolutePath().normalize(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static String join(List<String> values) {
        StringBuilder result = new StringBuilder();
        for (int index = 0; index < values.size(); index++) {
            if (index > 0) result.append('\t');
            result.append(safe(values.get(index)));
        }
        return result.toString();
    }

    private static String safe(String value) {
        return value == null ? "" : value.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ');
    }
}
