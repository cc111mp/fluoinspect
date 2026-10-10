# Validation of fluorescence QC methods

Software tests verify decoding, numerical behavior, coordinates and evidence
binding. They do not establish detection accuracy on biological images.

For core proposals, compare the intended target and outer envelope with reviewed
native-coordinate boundaries, including dark gaps, fragmented tissue and partial
neighbours. Coarse tissue support and geometric hulls are unreviewed proposals;
shape plausibility alone cannot verify core identity. Check that internal dim/zero
pixels remain in the outer envelope and that mask filling never modifies source
intensities. Challenge ambiguous pairs, incomplete cores, large one-sided losses,
elongated fragments and shadows. Preserve unresolved states instead of forcing
a complete circular core.

Compare region attribution using identical source-bound detector measurements
before changing thresholds or tile anchors. Account for interior, boundary/mixed,
nearby-background, neighbour and outside candidates. Exact area tests validate
the recorded coarse-label mapping, not biological boundary accuracy. Whole-source
detector coverage does not establish human/model inspection or artifact recall.
Source/mask rechecks must reject changed pixels, geometry and forged review/QC
claims. Existing expert image-level tags stay unchanged; localized correspondence
requires separate review fields. A lower candidate count is not an accuracy result.

Analysis-region receipts require retained receipt hashes, unchanged parent and
child sources, exact native crop readback and a checked child-to-parent mapping.
Rectangle envelopes retain internal dark values but do not establish biological
core identity. Compare target selection with expert-reviewed boundaries,
including incomplete cores and neighbouring tissue. Reevaluate thresholds when
crop support or foreground policy changes; dark-retaining envelopes may include
glass or physiological spaces and do not independently improve specificity.

Coverage planning must handle image edges, overlapping tiles, native sampling,
remaining view budgets and explicit omissions. Check area unions against an
independent raster on controlled geometries. Keep planned, exported and inspected
scope distinct. Native crops and preview resolution establish available evidence,
not successful model or human review. Validate inspection coverage against
expert-localized artifacts separately before enabling quality acceptance.

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

For alternating-band measurements, vary native angle, strip width, separation,
contrast, texture, sampling scale and region edges. Challenge long curved tissue
channels and localized dark holes alongside known straight bands. Both normal
tissue and technical errors may have parallel dark features: native intensity and
edge verification do not adjudicate their cause. The low-signal edge descriptors
are experimental, and their operating thresholds require independent validation.
Record per-orientation/scale and global proposal limits, unresolved fine widths,
sampling steps and missing flank context. A rectangle alone does not establish the
target-core identity. Include native crop/source binding in the final integration
evaluation, and compare actual inspected regions with expert-localized defects.

Original overlapping fields or independent references support geometric stitching
assessment. Before/after images and flat/dark-field references support correction
assessment. A dark hole or line in a final export cannot establish its cause alone.

For local observations, freeze tile geometry and native-unit settings before
comparing with reference annotations. Test isolated holes, non-repeating lines,
one-sided steps, smooth illumination gradients and natural cavities/boundaries.
Test partial edges and overlapping tiles against an independent raster; a full
evaluated footprint means every scheduled region ran through these bounded
methods, not that every artifact width/orientation was resolved. Include zero
candidate tiles in review exports. Record candidate omissions, native proposal
scale, original-pixel checks and uninspected scope separately. Report additional
reference cases routed and additional expert-OK workload, rather than treating
candidate routing as correct defect classification. Negative periodic and local
screens must retain unverified status even with a complete detector footprint.

After method validation, evaluate the complete agent on held-out cases. Compare
fixed views with adaptive investigation under documented budgets. Record actual
inspection coverage and model settings. A negative model response or detector
score must not become automatic acceptance of unexamined regions.
