# Validated v6: redistribute fuel requests with downstream capacity limits

Based on v3. After the existing LP and baseline blend, move at most half of
a lane's original request to another lane sharing its first edge, commodity
and destination fuel group. The aggregate request on that entry edge and from
that source remains unchanged. Nuclear flows retain the v3 baseline override.

A transfer is considered when the donor route has an observed closure or zero
capacity, is estimated to enter an edge after an announced prohibition starts,
cannot deliver before episode end, or has at least two extra weeks of queue
delay and total delivery time compared with the alternative.

The alternative must be currently allowed, have no such predicted restriction,
fit the remaining episode, and cost no more than 1.5 times the donor's observed
freight plus tariffs per unit. Transfer is limited by the recipient's nominal
request ceiling and its downstream capacity reservations. Notices are read only where all fields are observed, and are
not remembered when absent. Decoys remain possible; the 50% limit retains part
of the original route. Ordinary travel-time differences alone do not trigger
a move without queue delays or a restriction.

The routing overlay now estimates each exit's wait from its visible queue and
current edge capacity, using the larger of that delay and the existing pooled
delay. It reserves both edge and chokepoint-pool capacity for visible queued
and in-flight shipments, including their remaining downstream path. New requests
also share this reservation book across all goods and destinations. A transfer
updates reservations immediately, so a later transfer cannot reuse the same room.

Available service is current capacity times nominal travel time to the resource
plus one week, less committed cargo and estimated new dispatch. Added queue delay
does not create service credit. Dispatch estimates apply entry-edge sharing;
source stocks and fleet constraints may reduce execution further. All visible
inbound commitments count even if they would arrive later, making the guard
conservative. Resources common to donor and recipient keep the same aggregate
request. Nuclear requests are restored before building this reservation book,
so other shipments account for their final load. The LP inventory calculation
itself is unchanged from v3.

This version does not redirect cargo already at sea or in queues, or switch
between sources or different entry edges. Shared downstream throughput and
fleet constraints still apply, so preserved request totals do not imply equal
executed quantities or costs. Tests: `tests/test_routing_candidate.py`.
Frozen final v6 (`final/`), previous candidate (`old/`) and v3 (`v3/`):
`outputs/routing-v6/`. The intermediate `candidate/` predates the nuclear
reservation-order correction and is kept only for reproducibility. The two diagnosed
failures are regression scenarios; independent validation uses 64 Small episodes
at entropy 202610045 and 64 Full episodes at entropy 202610046.

## Final v6 results and promotion

| Task | Mean cost savings vs v3 | Paired 95% interval | Wins / losses / ties |
| --- | ---: | ---: | ---: |
| Small | +0.128689% | +0.080645% to +0.186200% | 51 / 8 / 5 |
| Full | +0.857331% | +0.635107% to +1.115734% | 55 / 5 / 4 |

Each row covers 64 paired scenarios. Metric: unweighted episode costs including
salvage, not RSS. Paired percentile bootstrap, 20,000 resamples, seed 202610045.
Against the previous routing candidate, savings are 0.120944% on Small (95%
interval includes zero) and 0.717445% on Full (95% interval +0.328783% to +1.085448%).
Known regressions: Small 1's extra cost vs v3 falls from 78.106 to 4.581 billion
USD; Full 29's extra cost falls from 391.897 billion to zero.

Dev Small RSS with CPU enforcement: 0.43941595 vs v3's 0.43700304; difference
+0.00241291, paired 90% interval [+0.00129072, +0.00366075]. No fallback weeks,
solver failures or measured CPU-budget violations. Final tests: 119 passed,
4 expected skips; isolated Tiny/Small/Full checks passed. The exact final source
was promoted to `agents/mine/agent.py`; v3 is preserved under `outputs/routing-v6/v3/`.

The nuclear reservation-order correction was made during the initial evaluation.
The final frozen version was rerun on all 128 scenarios; no parameters were tuned
against those results. See `outputs/routing-v6/summary.json`, `decision.json`,
`final-dev-small.json` and the Ukrainian explanation in `docs/MINE.md`.

## Previous v5 validation: positive mean, improvement not established

On 4 October 2026, frozen v3 and the previous v5 candidate ran 64 new Small episodes
(entropy 202610043) and 64 new Full episodes (entropy 202610044), 256 complete
episodes total. Each container had 1 CPU and 4 GB RAM.

| Task | Mean cost savings vs v3 | Paired 95% interval | Wins / losses / ties |
| --- | ---: | ---: | ---: |
| Small | +0.054185% | -0.085826% to +0.184695% | 32 / 31 / 1 |
| Full | +0.341838% | -0.011295% to +0.695264% | 41 / 22 / 1 |

Costs are unweighted episode costs including terminal salvage, not RSS.
Paired percentile bootstrap: 20,000 resamples, seed 20261004. Both intervals
include zero, failing the predeclared promotion gate. `mine` remains v3.
Rerouting occurred in 2,186 Small weeks and 4,943 Full weeks, across all 64
episodes in each task. No solver failures or measured CPU violations occurred
in either policy. The cost runner measures CPU but does not enforce fallback.

On the 20 existing dev Small episodes with CPU-budget enforcement, candidate
RSS was 0.43963445 versus v3's 0.43700304, difference +0.00263141 and paired
90% interval [-0.00679934, +0.01069016]. There were no fallback weeks or CPU
violations. This does not establish a score improvement either.

The full suite passed 96 tests, with 4 expected skips (optional RL and nested
Docker checks). An additional copy of all 18 v3 policy regression tests passed
against this candidate. Lint and isolated checks on Tiny, Small and Full passed.
No submission was uploaded. See `summary.json`, `dev-small.json`, `decision.json`
and the run/verification scripts under `outputs/routing-v5/`.
