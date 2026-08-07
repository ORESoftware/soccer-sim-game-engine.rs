# DEN-863: canonical typed soccer reward contract

`src/des/general/reward_contract.rs` defines the canonical version-1 component
schema used to score one soccer transition while retaining its raw contribution
breakdown.

## Sign convention

Penalties are negative rewards. Consumers that prefer a cost objective must use
`cost = -reward`; `RewardBreakdown` records both values so the conversion is
explicit and testable.

## Components

The version-1 contract separates:

- terminal match outcomes: goal for and goal against;
- signed shaping deltas: expected goals, possession value, progressive action,
  and defensive integrity;
- event penalties: foul, offside, and turnover;
- the infeasible-action safety penalty.

Missing runtime components contribute zero. Duplicate values, non-finite values,
and values outside their declared bounds are rejected instead of silently
clipped or double-counted.

## Anti-specification-gaming boundary

Contract validation requires:

- unique components and finite ordered bounds;
- reward components that cannot become negative;
- penalty components that cannot become positive;
- fixed symmetric goal-for/goal-against terminal rewards;
- a strictly negative optional infeasible-action penalty;
- terminal outcome dominance over the maximum absolute envelope of every
  non-terminal component plus a configured margin.

For `soccer-reward-v1`:

- terminal goal magnitude: `100`;
- maximum absolute non-terminal envelope: `18`;
- required terminal dominance margin: `20`;
- actual terminal dominance margin: `82`.

## Focused validation

The module uses only the Rust standard library and can be tested without
compiling the giant engine module:

```bash
rustc --edition=2021 --test src/des/general/reward_contract.rs \
  -o /tmp/reward-contract-tests
/tmp/reward-contract-tests
```

Engine integration should record `RewardBreakdown` before any normalization or
clipping and persist the contract schema/name in experiment manifests. Shaped
reward remains diagnostic and trainable signal; held-out win rate and
exploitability remain the deciding evaluation metrics.
