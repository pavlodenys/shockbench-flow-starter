# Experimental fuel and semiconductor MPC

Submission-safe prototype of a 24-week rolling LP across fuel, electricity,
wafer, fab WIP, OSAT packaging and market demand. It reads only the public
config and current observation. Nuclear actions and failure fallback retain
the established `mine` fuel policy copied into this self-contained submission.

The first week's automatic packaging and shared fab energy allocation are
constrained; future production, fuel loading and queue timing remain approximations.
`chip_blend=0.8`, `fuel_blend=0.55`; solve time limit 0.6 seconds.

Small dev: RSS **0.621016** versus previous mine **0.492662**; paired gain **0.128354**,
90% interval **[0.103052, 0.157787]**, 18 better / 2 worse scenarios.
No CPU violations or fallback weeks. These are local scores.

Independent Small root 202610073, 32 scenarios: RSS **0.600373** versus
previous mine **0.470603**, paired gain **0.129769**, 90% interval
**[0.105835, 0.155725]**; 30 better / 2 worse scenarios.
The RSS 0.7 target remains unmet.

Detailed diagnosis, assumptions, independent validation and next steps:
[SUPPLY_CHAIN_PLAN.md](../../docs/SUPPLY_CHAIN_PLAN.md).

Run `uv run sbf compare mine outputs/submissions/mine-before-chain-20261006 --task=small --episodes=dev --cpu_budget`.
Run `uv run sbf check chain_candidate --task=small` before adopting or packaging.
On 2026-10-06 this code and its params were promoted byte-for-byte to `mine`.
The user reported production RSS **0.60059**. The server report was not fetched again.
