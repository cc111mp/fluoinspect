# Typed QC decisions

FluoInspect separates observation, decision and action. The decision interface
accepts a structured state and named typed questions. A caller-supplied backend
returns bounded answers; ordinary code validates the response and decides what
review step is available. This follows the interface pattern described by
[TypeSafe](https://docs.typesafe.ai/introduction), without implementing Jev's
trained model, claiming its calibration, or depending on its service.

## Questions and answers

`agents.decisions` provides three primitives:

| Question | Answer contract |
| --- | --- |
| Choice | An option from declared criteria and a complete probability distribution |
| Score | A probability distribution over ordered integer rubric tiers; its expected value is the score |
| Binary | A finite probability that a specific proposition is true |

Questions define instructions and criteria before a model runs. Requests freeze
their structured state and criteria and compute a SHA-256 identity. Responses
must reference that identity and answer exactly the requested questions.
Validation rejects unknown labels, missing answers, extra fields, nonfinite or
out-of-range probabilities, contradictory choices and inconsistent scores.
Distributions must sum to one; invalid values are not clipped or normalized.

`DecisionBackend.decide` is a provider-independent asynchronous contract.
`ask_decisions` calls the supplied backend and validates its response. No hosted
client or default model is configured. An adapter must translate its provider's
output into FluoInspect's versioned response format; this is not wire-compatible
with the TypeSafe API.

## QC policy

`agents.triage.make_triage_request` creates three questions from an evidence packet:
the next review step, whether a provisional concern is supported, and human review
priority. Available choices are `human_review`, `unassessed`, and, when source
integrity and other conditions permit, `record_concern` or `gather_evidence`.
Sealed sessions and exhausted view budgets cannot offer additional crop requests.
Observation summaries can be supplied by recorded view ID; unknown view references
are rejected. A crop's existence alone does not establish a concern. Logging a
concern requires native evidence and supplied observations or measured axial-pattern
evidence, and still does not establish its biological or processing cause.

`apply_triage_policy` validates the response again, preserves unsupported checks,
and overrides routing when integrity or an optional operational confidence cutoff
requires escalation. It returns a recommendation; it executes no image operation.
The existing crop tools must recheck the live source before any later action.
Image acceptance and rejection are not offered by this initial policy, and human
QC decision fields remain blank.

Multiple concerns can coexist. A triage choice is a next step, not a mutually
exclusive artifact class. Report observations and their evidence separately.
Missing calibration or reference images do not become a negative finding.

## Confidence and validation

Reported confidence is a model output. It is not an established probability that
an AF or fluorescence QC judgment is correct. Receipts and recommendations mark
QC calibration and artifact accuracy as unvalidated. A confidence cutoff is an
operational review rule; it is not an experimentally established acceptance limit.

Evaluate candidate decision backends with fixed questions and independent
reviewed cases, alongside rules and constrained vision-language models using the
same evidence. Measure missed concerns, false flags, calibration, repeatability
and human review workload separately for each modality and artifact type.
Validate the detector evidence before relying on it in automatic decisions.

## Offline example

```bash
python examples/typed_triage.py
```

The example prints a request, validated synthetic response and recommendation.
It makes no model calls and measures no biological detection performance.
