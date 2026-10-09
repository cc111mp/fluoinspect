# Validation of fluorescence QC methods

Software tests verify decoding, numerical behavior, coordinates and evidence
binding. They do not establish detection accuracy on biological images.

For each detector, define the artifact or measured pattern, required references,
units and supported conditions. Build controlled challenges with known gain,
offset, transition width, orientation, displacement and noise. Include genuine
tissue boundaries and natural fluorescence variation as negative confounds.

Evaluate frozen settings on independently reviewed fluorescence and
autofluorescence cases. Keep related acquisition fields, exports and crops in
the same dataset group. Report recall, false flags, localization, confidence
intervals and failures separately by artifact type and severity. Calibration
and instrument metadata are required for physical-unit claims.

For brightness-pattern measurements, additionally vary the supplied period, cycle
coverage, region boundaries, sampling resolution and natural texture. Within-image
cycle prediction is a consistency measure, not an independent image-level test.
Numerical tests allow documented floating-point tolerance under axis transposition.
Period selection, biological confounds and artifact thresholds require independent
validation. Broad extrema are useful context proposals even when their cause is
uncertain; unresolved checks must not become negative findings or acceptance.

Original overlapping fields or independent references support geometric stitching
assessment. Before/after images and flat/dark-field references support correction
assessment. A dark hole or line in a final export cannot establish its cause alone.

After method validation, evaluate the complete agent on held-out cases. Compare
fixed views with adaptive investigation under documented budgets. Record actual
inspection coverage and model settings. A negative model response or detector
score must not become automatic acceptance of unexamined regions.
