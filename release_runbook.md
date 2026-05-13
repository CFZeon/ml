# Capital Release Runbook

This runbook describes how deployment readiness progresses from research certification to capital-bearing stages.

## Stages

1. `research_only`
   - no approved champion is available
   - no deployment action is permitted

2. `research_certified`
   - an approved champion exists
   - historical certification is complete
   - no live capital is permitted

3. `paper_verified`
   - paper or shadow-live calibration is green
   - the model may remain in paper monitoring
   - no live capital is permitted yet

4. `micro_capital`
   - paper verification is green
   - operational limits are healthy
   - kill switch is ready
   - current drawdown remains above the configured kill-switch threshold
   - an operator has manually acknowledged the micro-capital release

5. `scaled_capital`
   - all micro-capital requirements still pass
   - an operator has explicitly approved scale-up

## Advancement Rules

Advance from `research_certified` to `paper_verified` only after a green paper or shadow-live calibration report is attached.

Advance beyond `research_certified` only after a validated deployment profile is selected. Readiness now blocks paper and capital stages unless the monitoring report carries one explicit operating-envelope profile plus replay telemetry proving the target hardware budget.

No stage should advance when the current champion is past its freshness TTL. By default, deployment readiness expires a promoted champion after 28 days and surfaces `model_expired` until a fresh version is retrained, validated, and promoted.

Advance from `paper_verified` to `micro_capital` only when all of the following are true:

- deployment readiness is otherwise healthy
- operational limits report `healthy=True`
- operational limits report `kill_switch_ready=True`
- operational limits report `drawdown_breached=False`
- the release request includes `manual_acknowledged=True`

Advance from `micro_capital` to `scaled_capital` only when all micro-capital controls remain green and the release request includes `scaled_capital_approved=True`.

## Operating Envelope

Select one explicit deployment profile through `deployment.profile`, `monitoring.deployment_profile`, or the readiness policy before expecting paper or capital stages to advance.

| profile | max stage | peak RSS | model load | inference p95 | drift cycle | storage | max symbols | max timeframes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `research_workstation` | `scaled_capital` | 16384 MB | 2000 ms | 500 ms | 900000 ms | 40960 MB | 64 | 12 |
| `consumer_laptop` | `micro_capital` | 8192 MB | 3000 ms | 1000 ms | 1800000 ms | 20480 MB | 16 | 6 |
| `mini_server` | `scaled_capital` | 12288 MB | 1500 ms | 400 ms | 900000 ms | 30720 MB | 32 | 8 |
| `reduced_power_research_only` | `research_certified` | 4096 MB | 5000 ms | 2500 ms | 3600000 ms | 10240 MB | 4 | 2 |

`reduced_power_research_only` is never capital-eligible. When the operating envelope is breached, the degraded-mode plan is to fall back to simpler detectors, reduce the candidate set, disable mixture routing, and downgrade the runtime back to research-only.

The replay benchmark attached to the monitoring report should record:

- throughput in bars per second
- latency tail percentiles
- memory spikes
- restart recovery time
- degraded-mode behavior

## Required Inputs

`build_deployment_readiness_report(...)` should be called with:

- `paper_report` for paper or shadow-live verification
- `operational_limits` for kill-switch and risk-limit readiness
- `release_request` containing the requested stage and any manual approvals
- `policy` when you need to override the default 28-day champion TTL through `max_model_age_days`, `warn_model_age_days`, or `as_of_timestamp`, or when you want to select `deployment_profile`
- a monitoring report that includes `operating_envelope.resource_telemetry` and `operating_envelope.replay_benchmark` for any stage above `research_certified`

`build_operational_limits_report(...)` can be used to normalize kill-switch readiness plus current equity drawdown into the `operational_limits` payload. The default drawdown ceiling is 10% unless a stricter policy overrides it.

## Blocking Signals

Always inspect `release_blockers` before acting on readiness output.

Common blockers include:

- `paper_verification_required`
- `deployment_profile_unselected`
- `deployment_profile_stage_exceeded`
- `replay_benchmark_unavailable`
- `manual_ack_required_for_micro_capital`
- `model_expired`
- `operational_limits_unavailable`
- `kill_switch_not_ready`
- `drawdown_limit_breached`
- `kill_switch_triggered`
- `scaled_capital_approval_required`
- backend, monitoring, drift, or rollback failures propagated from the readiness components

## Operator Actions

- `hold`: do not advance the release stage
- `certify`: certification is complete but the system is not yet paper-verified
- `paper`: the system is ready for paper or shadow-live verification, not live capital
- `deploy`: the requested capital stage is eligible