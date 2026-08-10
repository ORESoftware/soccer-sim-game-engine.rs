# DEN-862: canonical exploration trace library

`src/des/general/exploration_trace.rs` is the reusable behavior-policy boundary
for the canonical soccer engine. Tactical scoring and feasibility remain in the
existing engine; this module owns versioned profiles, deterministic sampling,
and selected-versus-executed likelihood metadata.

## Supported profiles

- greedy;
- epsilon-greedy;
- configurable rank-weighted sampling, including 70/20/10 experiments;
- Boltzmann sampling;
- uncertainty-directed softmax.

Training and adaptation modes may sample stochastically. Validation and
production profiles are forced to greedy behavior even when a stochastic
strategy was configured accidentally.

## Downstream overrides

The selected action initially has an exact behavior probability and
log-probability. After the authoritative rules, feasibility, or MPC layer applies
an action, callers must invoke `reconcile_executed_action`.

When the executed action differs from the selected action, the trace marks its
likelihood as `UnknownAfterDownstreamOverride` and clears the executed
probability/log-probability. Reusing the selected action probability for a
changed action would fabricate on-policy data and can corrupt PPO or other
importance-weighted updates.

## Focused validation

The module uses only the Rust standard library, so it can be tested independently
of the multi-megabyte engine and external workspace dependencies:

```bash
rustc --edition=2021 --test src/des/general/exploration_trace.rs \
  -o /tmp/exploration-trace-tests
/tmp/exploration-trace-tests
```

Production integration should map existing tactical candidates into
`ActionCandidate`, persist `ActionSelectionTrace` with each transition, and
reject or separately model on-policy samples whose executed likelihood is
unknown.
