# Detector development workflow

Start with one fixed scoring command that joins saved measurements to immutable
expert annotations by image identity. Freeze the cohort, label definitions and
baseline predictions; report disagreements and unassessed cases explicitly.
Keep images, workbooks, annotation rows, run results and institution-specific
scoring adapters outside the public repository.

1. Reproduce the baseline counts before detector changes. Keep related exports,
   crops and acquisition fields in the same acquisition group.
2. Prepare disagreement panels with a whole-image overview, tile-level map and
   native detail crops. Share the linear display window within each panel and
   record clipping. Leave reviewer verdicts blank until an expert reviews them.
3. Distinguish target-core tissue, neighbouring tissue and surrounding background.
   Brightness-derived support does not verify a core identity. Spot-only crops
   remain background/detail observations rather than representative tissue views.
4. Explain misses and false flags before selecting features. Freeze measurements,
   then fit each threshold/model on all but one acquisition group and score the
   excluded group. Report per-group results as well as pooled recall, false flags
   and unassessed rates.
5. Measure before/after performance using the same scorer. If feature development
   used all groups, holding out only thresholds does not make that study an
   independent validation. Test the frozen detector on new acquisitions.

Human reference labels are separate from detector inputs and reviewer corrections
are separate from original annotations. A negative periodic or local screen
remains unverified. Detection-method validation precedes automated agent routing
and final acceptance policies; changing preview contrast is not a detector gain.
