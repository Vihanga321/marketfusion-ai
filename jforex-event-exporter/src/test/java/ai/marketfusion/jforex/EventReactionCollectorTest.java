package ai.marketfusion.jforex;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class EventReactionCollectorTest {
    @Rule
    public TemporaryFolder temporary = new TemporaryFolder();

    private Path writeEligible(String changedStatus) throws IOException {
        Path path = temporary.newFile("eligible.tsv").toPath();
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("event_id\tevent_type\treference_period\tevent_timestamp_utc\t"
                    + "production_status\tadjudication_class\tmodel_eligible_market_reaction\t"
                    + "validator_version\treaction_contract_version");
            writer.newLine();
            for (int index = 0; index < EventReactionCollector.EXPECTED_EVENTS; index++) {
                String status = index == 0 && changedStatus != null ? changedStatus : "PASS";
                writer.write("event-" + index + "\tus_cpi_release\t2024-01-01T00:00:00Z\t"
                        + Instant.ofEpochSecond(1_700_000_040L + index * 60L).toString() + "\t"
                        + status + "\tSTRICT_PASS\tTrue\t"
                        + EventReactionCollector.VALIDATOR_CONTRACT + "\t"
                        + EventReactionCollector.REACTION_CONTRACT);
                writer.newLine();
            }
        }
        return path;
    }

    @Test
    public void exactStrictPassUniverseIsAccepted() throws Exception {
        List<EventReactionCollector.EventRow> rows = EventReactionCollector.readEligible(writeEligible(null));
        assertEquals(242, rows.size());
    }

    @Test(expected = IOException.class)
    public void quarantinedStatusCannotEnterCollector() throws Exception {
        EventReactionCollector.readEligible(writeEligible("MISMATCH"));
    }
}
