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
   - an operator has manually acknowledged the micro-capital release

5. `scaled_capital`
   - all micro-capital requirements still pass
   - an operator has explicitly approved scale-up

## Advancement Rules

Advance from `research_certified` to `paper_verified` only after a green paper or shadow-live calibration report is attached.

Advance from `paper_verified` to `micro_capital` only when all of the following are true:

- deployment readiness is otherwise healthy
- operational limits report `healthy=True`
- operational limits report `kill_switch_ready=True`
- the release request includes `manual_acknowledged=True`

Advance from `micro_capital` to `scaled_capital` only when all micro-capital controls remain green and the release request includes `scaled_capital_approved=True`.

## Required Inputs

`build_deployment_readiness_report(...)` should be called with:

- `paper_report` for paper or shadow-live verification
- `operational_limits` for kill-switch and risk-limit readiness
- `release_request` containing the requested stage and any manual approvals

## Blocking Signals

Always inspect `release_blockers` before acting on readiness output.

Common blockers include:

- `paper_verification_required`
- `manual_ack_required_for_micro_capital`
- `operational_limits_unavailable`
- `kill_switch_not_ready`
- `scaled_capital_approval_required`
- backend, monitoring, drift, or rollback failures propagated from the readiness components

## Operator Actions

- `hold`: do not advance the release stage
- `certify`: certification is complete but the system is not yet paper-verified
- `paper`: the system is ready for paper or shadow-live verification, not live capital
- `deploy`: the requested capital stage is eligible