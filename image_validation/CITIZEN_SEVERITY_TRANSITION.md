# Citizen-reported severity transition — October 27, 2026

Branch: `feature/citizen-reported-severity` (models and companion backend repository).

## Implemented locally

- Keep image validity decisions and issue-category classification.
- Stop loading severity text prompts and stop image scale/obstruction inference in both inference services.
- Retain the six legacy severity response fields as null for API compatibility.
- The companion backend ignores image severity, including predictions from an older deployed classifier, preserving citizen scale/obstruction for the existing severity calculation.
- Preserve experiments SEV-001–016, label snapshots, and evaluation artifacts. No further training or re-labeling is required for this transition.
- Existing severity weights, category hazard baselines, persistence, and location factors still apply. This change does not claim citizen-input accuracy or produce a new numeric trust score.

## Agreed direction for the next implementation

The user supplied an earlier proposal combining an AI severity gate, citizen reputation, and officer approval. The latest instruction explicitly disables image severity, so the AI gate is superseded.

1. Citizens supply report severity inputs and may request urgent verification with an event/obstruction reason.
2. An urgency request enters **Pending Urgent Verification**. It does not automatically escalate dispatch or promise same-day service.
3. An authorized officer approves urgency, declines it, or explicitly records a proven false urgency claim with a reason.
4. Trust tracks verified reporting reliability. A declined request, low-severity report, model rejection, or ordinary mistake is not automatically dishonesty.
5. Confirmed abuse can reduce trust and restrict future urgency flags; ordinary environmental reporting remains available.
6. Keep trust separate from incident severity. New or low-trust citizens can still report a serious incident for officer review.

## Remaining work — not implemented by the disablement change

- Define numeric trust initialization, updates, restriction threshold, recovery/appeal, and who can reverse a decision. The supplied proposal contains no numeric policy.
- Add durable urgency requests and auditable officer decisions, enforce permissions and idempotent score updates on the server.
- Add citizen request and officer triage UI, then connect only approved urgency to scheduling.
- Deploy the backend protection and model image; no deployment or historical report recomputation has been performed here.

Validation: inference contract tests cover approved, manual-review, and rejected images; backend tests prove old image-severity predictions cannot reach the report aggregation boundary.
