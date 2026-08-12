# SQL Data Feed — Plan

**Status:** ⏸ Parked (deferred for later, after Partner SDK + Widgets ship)
**Date:** 2026-08-12
**Owner:** siridech.bo@kmitl.ac.th
**Related tracker item:** to be added under OPEN ITEMS as `F5` when unparked

---

## Why this exists (customer context)

Customer runs a Mitsubishi MES (Manufacturing Execution System) that collects data
from PLCs on the factory floor and writes it to a **Microsoft SQL Server** database.
Today, to use CiRA ME with that data, they have to manually export SELECT results
to CSV and upload. They want CiRA ME to query the SQL server directly.

**Sample data reviewed:** `D:\tmp\OEE_202607.xlsx` (2026-08-12) — one month of
data from one production line, 109k rows across two related tables:

- `OEE_202607` — machine status stream (Machine, Status, Running, Timestamp,
  LineStatus, etc.). Wide format, event-based (irregular timestamps),
  mixed categorical + numeric.
- `vw_RejectFeeder` — defect event log (MachineName, AddDate, Detail with
  Thai text like "ID ใหญ่", QTY). Used as labels via JOIN.

## Scope (reduced — Phase 1 only)

**In scope for the first ship:**

1. New "SQL Server" data source in the Dataset section
2. Connection management — user saves MS SQL Server connections once
   (host, port, database, auth), reuses across queries
3. Raw SELECT editor with a "Preview first 100 rows" button
4. Ingest — full query result is streamed to CSV on disk, then loaded through
   the existing CSV → data session pipeline (no new pipeline code)
5. Auto-encode categorical columns (Status='AUTO'/'WAIT' → one-hot or label)
6. Missing value handling — treat `'NULL'`, `'null'`, `''`, `'NA'` as missing
7. MS SQL Server driver (`pyodbc`) with support for both SQL auth and
   Windows Authentication (factories often use the latter)
8. UTF-8 encoding on the connection (Thai text must survive)

**Explicitly deferred (later phases, separate plans if greenlit):**

- **Phase 2 — Daily refresh scheduler.** ID/timestamp watermark, cron trigger,
  append-only ingest. Real value but adds scheduling infrastructure.
- **Phase 3 — Live SQL polling** as an App Builder input node
  (`input.sql_poll`). Parallel to MQTT input. Enables real-time inference on
  SQL-backed data.
- **Time-based windowing primitive** (`window = "last N minutes"` vs. today's
  row-count windowing). Necessary for irregular event data. Belongs in the
  pipeline, not in the SQL feed itself — but it's a prerequisite for really
  useful MES models. Track separately.
- **Auto-retrain / MLOps loop.** Explicitly out — separate product conversation.
- **PostgreSQL / MySQL / Oracle drivers.** MS SQL Server first; customer isn't
  asking for others yet. Add later on demand.

## Design decisions (locked from the 2026-08-12 discussion)

- **All 3 prediction targets supported** (downtime, defect, OEE regression).
  No SQL-side logic — user runs 3 separate CiRA ME projects with 3 different
  saved queries (one per target).
- **Both per-machine and cross-machine models supported.** User picks in SQL
  (`WHERE Machine = 'B079'` for per-machine, no filter for cross-machine).
  Machine column auto-encoded as a categorical feature when kept.
- **Time granularity: customer doesn't know yet.** Defer this decision to the
  time-windowing primitive (deferred to a later phase). For Phase 1, we ingest
  raw rows and let existing row-count windowing handle it — good enough for
  proof-of-concept models.
- **LAN-accessible SQL Server** confirmed. No VPN / tunnel plumbing needed.
- **Watermark strategy for daily refresh (Phase 2):** ID-based (monotonic)
  preferred over Timestamp-based (guards against late-backfill).

## Known gotchas surfaced by the OEE file

1. **Mixed data types** — Machine (string), Status (string), Running (int).
   Auto-encoding required.
2. **`'NULL'` as literal string** — someone exported that way. Must be treated
   as missing.
3. **Irregular timestamps** — 08:46:30 → 08:56:23 → 08:59:32 — event-based, not
   periodic sampling. Pipeline row-count windowing works but isn't
   semantically ideal. Time-based windowing (deferred) is the real fix.
4. **Multi-machine mixed in one table** — cross-machine model needs Machine
   auto-encoded; per-machine model needs user to filter.
5. **UTF-8 / Thai encoding** — pyodbc default codepage is Windows-1252.
   Need explicit `charset=utf8` or ODBC connection string tweak.
6. **Data volume** — 109k rows / month / line. A year of 5 lines ≈ 7M rows.
   Manageable if we stream to CSV rather than materialize in memory.

## Milestones (when unparked)

| # | Milestone | Effort |
|---|---|---|
| 1 | MS SQL Server connection + preview (SQL editor + first-100-rows table) | 3-4d |
| 2 | Full ingest → CSV → existing data session | 1-2d |
| 3 | Auto-encode categoricals + NULL string handling | 2-3d |
| 4 | Windows Authentication + UTF-8 encoding tests against real MS SQL | 2d |
| 5 | Connection storage (encrypted) + Data Connections page | 2-3d |
| 6 | Documentation + example queries against the OEE file | 1d |
| **Total** | Phase 1 shippable | **~2 weeks** |

## Open questions (to resolve with customer when unparking)

1. **OEE score formula** — do they compute it live in SQL, or is there a
   pre-computed OEE-per-hour table already? Second option is much cleaner
   for regression labels.
2. **Windows Auth vs SQL auth** — which does their DBA prefer? Windows Auth
   is fiddly on Linux backends (needs Kerberos + `krb5.conf`).
3. **Read-only DB user** — will they create a dedicated `cirame_read` account,
   or expect us to work with their existing user?
4. **Multi-machine data ergonomics** — when a query returns 5 machines, do
   they want us to auto-split into 5 datasets, or keep it as one dataset
   with Machine as a feature? Design choice.

## Test fixture (built-in)

The `D:\tmp\OEE_202607.xlsx` file is the perfect end-to-end test dataset.
When we build this, wire it into the test suite via `pandas.read_excel`
and pretend it came from SQL — validates the whole ingest → encode →
pipeline path against real MES data without needing a live SQL server.

## Handoff to whoever picks this up

- This plan is deliberately parked. Do NOT start building until user says
  "unpark F5" or equivalent.
- When unparking, revisit this doc — customer's answers to the open questions
  may reshape the scope.
- Partner SDK and App Builder widgets (companion plans dated 2026-08-12) are
  the active workstreams. This one waits.
