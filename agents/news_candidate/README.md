# Candidate: prepare fuel stocks for announced prohibitions

This experimental policy copies v3 and adds a bounded reserve based on visible
`pending_prohibitions` entries. It does not read `messages` or `warning.score`.
Even legal publications can be decoys in the standard observation regime.

For a fuel supply group, the additional reserve is three weeks multiplied by
the share of currently available route capacity exposed to a prohibition
announced to start within eight weeks. A route contributes only if estimated
travel and queue times allow entry to its threatened edges before the stated
date and delivery within the episode. Duplicate entries do not multiply risk.
The added reserve changes both the order ceiling and LP shortage targets.

This is a heuristic: capacities shared by several routes and future queue
changes are not predicted exactly. The policy does not cancel existing cargo,
block routes early, or remember notices after they disappear from observations.
Nuclear requests retain v3's baseline override. Without relevant visible news,
the policy produces the same actions as v3 on the same state.

Validation artifacts: `outputs/news-v4/`. The reference v3 is frozen there;
`agents/mine` remains the accepted agent until the candidate passes the recorded
acceptance criteria. Tests are in `tests/test_news_candidate.py`.

## Result: not promoted (4 October 2026)

Paired comparison against frozen v3, 64 episodes each on new Small root
202610041 and Full root 202610042 (256 complete episodes):

| Task | Mean cost savings vs v3 | Paired 95% interval | Wins / losses / ties |
| --- | ---: | ---: | ---: |
| Small | -0.0000000389% | -0.0000001249% to 0% | 0 / 1 / 63 |
| Full | +0.0122853% | +0.0000506% to +0.0366931% | 7 / 5 / 52 |

These are unweighted episode costs including terminal salvage, not RSS.
Intervals use 20,000 paired percentile bootstrap resamples, seed 20261004.
The predeclared gate required a positive lower interval bound on both tasks;
it failed on Small. About 90% of Full's net savings came from episode 57.
The Full gain is small and concentrated, and needs independent replication.

The rule activated in 31 Small weeks (6 episodes) and 113 Full weeks
(20 episodes). Small same-state tracing found only two weeks with changed
orders; in 14 news weeks all affected groups already requested nominal maxima.
The sole Small cost difference was an increase of USD 75,822.96 in episode 20.

Validation: 83 tests passed, 4 skipped (optional RL dependency and nested Docker
daemon); lint and isolated `sbf check` runs passed for Tiny, Small and Full.
No LP failures or measured CPU-budget violations in the paired runs.
The diagnostic runner measures CPU but does not enforce server fallback.
`agents/mine/agent.py` remains v3. Nothing was uploaded.

On the 20 existing dev Small episodes, candidate RSS was 0.43698373 versus
v3's 0.43700304: difference -0.00001931, paired 90% interval
[-0.00004939, 0]. Neither agent had fallback weeks or CPU violations.
This additional check also provides no evidence of an improvement on Small.
