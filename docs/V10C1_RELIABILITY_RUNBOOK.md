# MarketFusion V1.0C.1 Reliability Runbook

## Purpose

V1.0C.1 is the clean dedicated-PC rerun of the frozen XAUUSD forward-validation protocol. The prediction hypotheses, targets, feature sets, model families and evaluation gates stay frozen. This layer improves only operational reliability.

Non-negotiable safety state:

- Automatic execution: `DISABLED`
- Runtime: `SHADOW_ADVISORY_ONLY`
- Manual confirmation: `REQUIRED`
- Forward backfill: `PROHIBITED`
- Retraining during forward window: `DISABLED`
- Production integration: `NOT ENABLED`

## Before migration

1. Stop the old V1.0C collector with `Ctrl+C`.
2. Confirm no `v10c_xauusd_forward_validation` or `v10c1_reliability` process is running.
3. Keep the old run; do not delete it.
4. The reliability-prep branch must pass CI before it becomes the dedicated-PC candidate.

## Saturday: archive and transfer

On the old machine, after checking out the reliability-prep candidate:

```powershell
.\scripts\prepare_v10c1_clean_start.ps1 -ConfirmArchive
.\scripts\v10c1_preflight.ps1 -RequireUninitialized
```

The preparation command moves the previous V1.0C forward directory and frozen-model directory into timestamped archive locations. It does not delete the old observations.

Transfer or clone the repository onto the dedicated PC and install the required Python, Node.js and MT5 environment. Do not transfer the archived observations back into the canonical forward path.

## Sunday: read-only dry run

Keep the canonical forward path uninitialized and run:

```powershell
.\scripts\v10c1_preflight.ps1 -RequireUninitialized
```

Expected checks include:

- MT5 Python package available
- MT5 terminal connected
- XAUUSD broker symbol discoverable
- XAGUSD, USDJPY and EURUSD frozen context identities ready
- broker-clock calibration PASS
- at least 5 GB free disk
- no existing V1.0C canonical manifest

This preflight does not record predictions or outcomes.

## Monday: official clean forward start

Only when the market is live and the final read-only preflight is PASS:

```powershell
.\scripts\run_v10c_xauusd_forward_validation.ps1 -Initialize
```

Run `-Initialize` exactly once.

Immediately verify:

```powershell
.\scripts\v10c1_preflight.ps1
```

Then start the reliable supervisor:

```powershell
.\scripts\run_v10c1_reliable_collector.ps1 -IntervalSeconds 60 -MaxRetrySeconds 300
```

Do not run `-Initialize` again.

## Reliable supervisor behavior

A healthy cycle calls the unchanged frozen V1.0C one-cycle logic. If MT5 IPC, broker clock, context history or another provider dependency fails, the supervisor:

1. fails closed and records no prediction for that failed cycle;
2. records an outage event;
3. shuts down the failed MT5 Python connection;
4. waits and reconnects;
5. retries on a later cycle;
6. never reconstructs a missed forward prediction.

The existing frozen 10-minute recording-delay rule remains authoritative. The supervisor does not expand it.

## Status

```powershell
.\scripts\run_v10c1_reliable_collector.ps1 -Status
```

Health data is written to:

```text
data\runtime\v10c1\collector_health.json
data\runtime\v10c1\provider_outages.jsonl
```

The full read-only dashboard API exposes the same data at:

```text
GET /api/system/reliability?symbol=XAUUSD
```

## Reboot recovery

After the official one-time initialization has completed, an optional Windows logon recovery task can be installed:

```powershell
.\scripts\install_v10c1_startup_task.ps1 -ConfirmInstall
```

The task restarts the reliable collector after user logon and unexpected process exit. The Python single-instance lock prevents duplicate collectors.

## Backup

Create a timestamped backup without changing the forward ledger:

```powershell
.\scripts\backup_v10c1_state.ps1
```

The backup contains SHA256 inventory for copied state files.

## Completion minimums

Each frozen contract must reach at least:

- 20 active trading dates
- 500 matured observations
- 400 directional outcomes
- 4 chronological stability blocks
- at least 80 directional rows per block

Only after minimum sample readiness are the frozen evidence gates evaluated. A pass does not automatically promote a model or enable production trading.
