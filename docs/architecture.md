# FluoInspect architecture

FluoInspect separates native image access, quantitative methods, investigation,
evidence and model integration. Both fluorescence and autofluorescence use the
same pixel-preservation contract; assay intent belongs in modality and channel
metadata and validated measurement profiles.

The `io` module reads supported uint16 TIFF planes. `detectors` proposes sharp
axial intensity patterns, while `measurements` returns descriptive statistics.
`pipeline.measure_image` combines these without assigning image acceptance.
The experimental brightness module records broad axis-profile extrema and
repetition at an explicitly supplied native-pixel period. It returns source
geometry, region/sampling eligibility and alternative-period comparisons without
assigning a causal artifact class. Its settings participate in the run identity,
and source-bound evidence packets retain the measurements or explicit absence for
legacy records. Context strips describe profile extrema, not confirmed seams.
Modality labels do not change or validate the development detector thresholds.

`segmentation` proposes coarse core envelopes, separate intensity-derived tissue
support and nearby background excluding provisional neighbours. It retains source
values, explicit native-cell mappings and unreviewed/unresolved identity. Optional
local candidate attribution uses unchanged source-bound measurements and tile
context; it never paints excluded pixels black before detection. Dedicated source,
mask and recomputed-geometry verification supports future tool integration.
The core CLI produces JSON and review CSVs; biological boundary accuracy and
agent consumption of this new evidence still require evaluation.

`investigation` stores bounded views and precise source mappings. Local ownership
locks serialize writes, identified retries reuse committed views, and sealed
reports block additional mutations. These locks are local filesystem locks,
not distributed job leases.

`tools.evidence` binds views and optional cached measurements to the same source.
Its checks establish evidence integrity, not artifact recognition. `tools.registry`
describes implemented method scope and methods still required. `agents` contains
fixed role briefs and a backend interface; it does not contain a functioning
inference client or autonomous investigation controller.

`deployment` creates reproducible whole-image assignments and checks local paths
and dependency metadata. One controller per node is the initial operating policy;
node-wide scheduling, automatic transfers and result publication are separate work.

The public repository extracts portable components from the development toolkit.
The original institution-specific batch runner and private run publications remain
outside this package. Future CSV batch reporting and a review application should
consume the public measurement and evidence contracts.
