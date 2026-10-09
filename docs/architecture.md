# FluoInspect architecture

FluoInspect separates native image access, quantitative methods, investigation,
evidence and model integration. Both fluorescence and autofluorescence use the
same pixel-preservation contract; assay intent belongs in modality and channel
metadata and validated measurement profiles.

The `io` module reads supported uint16 TIFF planes. `detectors` proposes sharp
axial intensity patterns, while `measurements` returns descriptive statistics.
`pipeline.measure_image` combines these without assigning image acceptance.
Modality labels do not change or validate the development detector thresholds.

`investigation` stores bounded views and precise source mappings. Local ownership
locks serialize writes, identified retries reuse committed views, and sealed
reports block additional mutations. These locks are local filesystem locks,
not distributed job leases.

`tools.evidence` binds views and optional cached measurements to the same source.
Its checks establish evidence integrity, not artifact recognition. `tools.registry`
describes implemented method scope and methods still required. `agents` contains
fixed role briefs, vision and decision backend interfaces, typed question/answer
contracts and code-owned triage routing. It does not contain a functioning
inference client or autonomous investigation controller. See
[typed decisions](typed-decisions.md) for evidence binding and confidence limits.

`deployment` creates reproducible whole-image assignments and checks local paths
and dependency metadata. One controller per node is the initial operating policy;
node-wide scheduling, automatic transfers and result publication are separate work.

The public repository extracts portable components from the development toolkit.
The original institution-specific batch runner and private run publications remain
outside this package. Future CSV batch reporting and a review application should
consume the public measurement and evidence contracts.
