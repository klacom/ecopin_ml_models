# Forward hotspot forecasting

Start with `python inference_service.py`; the existing container entry point and
`POST /feedback` stay available. Install `requirements.txt` before starting.
`POST /forecast/project` accepts `{"horizon":"30d","zone_type":"h3"}`.
Horizons are `7d`, `30d`, and `90d`. Configured `barangay` zones are also supported.
Legacy `/forecast` maps `daily`/`weekly` to `7d`, and `monthly` to `30d`.

The service reads `spatial_zones` and the validated-only
`zone_weekly_report_history` view using server-side Supabase credentials. Deploy
the forward-zone database migration first. It never falls back to unvalidated
reports. The default zone catalog is an H3 grid, not official barangay boundaries.

Each zone has 52 zero-filled, complete UTC Monday weeks. The current incomplete
week is excluded. A damped additive Holt trend projects a common 14-week trajectory;
exact overlaps with the next 7/30/90 days prorate counts within weeks. The baseline
is an equally long historical period ending at the last complete week. This
assumes reports are distributed uniformly within each weekly bucket.

The 95% prediction corridors are approximate innovation-covariance intervals with
a Poisson count-variance floor. Sparse data or failed optimization uses a pooled
mean and Gamma-Poisson predictive intervals, including uncertainty for no reports.
Intervals do not include estimated-parameter uncertainty for Holt fits and have
not been calibrated against held-out real observations. Longer horizons may be
equal for zero-rate or declining zones; positive forecasts increase cumulatively.

The specification's prose defines high risk as **more than 15** predicted reports
or **more than 40%** growth, and medium risk as 5–15 reports. Those rules take
precedence over its conflicting example code. `risk_score` is a bounded 0–1 risk
index, not an estimated probability.

Run deterministic model, HTTP, pagination, and 30-zone performance checks with
`python -m unittest discover -p "test_*.py" -v` in this directory.
`python smoke_test.py` additionally checks the configured migrated database.

## PDT-19: protected historical analysis

`POST /patterns/analyze` accepts `{}` and requires `Authorization: Bearer <SPATIAL_ANALYTICS_TOKEN>`. Configure the same server-only random secret on Node and Python. Missing credentials fail closed; do not expose the token in client environments. The service-role-only `read_hotspot_pattern_history(5000)` SQL RPC supplies one consistent snapshot after PDT-08 reconciliation and the PDT-19 migration.

Past 7/30/90-day observations run the shared haversine DBSCAN engine. Ninety-day groups retain non-overlapping 30-day recurrence evidence; shorter-window groups link by member identities, and barangay links use actual reconciled assignments. Raw report identities and point lists are never returned. The same snapshot supplies 52 complete UTC weeks to existing forward barangay forecasts for +7/+30/+90 days. Historical labels are transparent heuristics, not learned future spatial predictions or p-values. A single-analysis semaphore and a 5,000-point cap prevent unbounded concurrent DBSCAN jobs; oversized histories fail rather than sample.

Full rules, data exclusions, authentication, rollout dependencies and verification limits: [historical hotspot integration](../../ecopin-web/docs/historical-hotspot-patterns.md). The standalone `make_pattern_fixture.py OUTPUT_PATH` creates invented data only for workspace browser tests; it requires the adjacent backend roster and does not query Supabase. Run all tests from the ML repository root with `python -m unittest discover -s hotspot_forecasting -p "test_*.py"`.
