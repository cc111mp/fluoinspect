# FluoInspect

[![Tests](https://github.com/cc111mp/fluoinspect/actions/workflows/tests.yml/badge.svg)](https://github.com/cc111mp/fluoinspect/actions/workflows/tests.yml)

FluoInspect is a development toolkit for fluorescence and autofluorescence image
QC. It preserves native pixels, produces quantitative screening evidence, and
supports investigation from whole-image context to native detail crops.

The current release is **pre-alpha**. Image integrity, coordinates and numerical
behavior are tested. Artifact-detection accuracy and acceptance thresholds require
independent validation for each assay. Agent role instructions and a backend
interface exist; automated model inference and orchestration remain planned.

## Current scope

| Component | Available behavior |
| --- | --- |
| Native TIFF reading | One top-left, minisblack uint16 grayscale plane; segment and memory guards |
| Measurements | Intensity distributions, zero/storage-ceiling fractions, relative detail and provisional background variation |
| Brightness patterns | Experimental broad axis-profile differences and repetition at an explicitly supplied native-pixel period; no quality labels |
| Alternating bands | Opt-in oriented dark-band and parallel-pair measurements, sampled original-pixel checks and straight-edge/zero-pixel descriptors; no quality labels |
| Axial screening | Sharp horizontal/vertical intensity-pattern candidates; not causal stitching labels |
| Investigation | Overview, context, detail and unlabelled comparison views with source coordinates |
| Analysis regions | Exact native rectangle crops, checked parent mapping and opt-in envelope measurements retaining dim/zero pixels; core identity remains unreviewed |
| Coverage | Bounded systematic detail plans and exact unions of exported native/preview areas; inspection remains unestablished |
| Local artifact observations | Systematic tile evaluation of dark regions and abrupt axial steps, including unflagged areas; original-pixel rectangle checks and an evaluated-footprint ledger |
| Review routing | Negative, unassessed and candidate results remain unverified; separate profile decisions stay pending |
| Evidence | Source hashes, exact native crop readback, record binding and explicit unassessed checks |
| Deployment | Fixed job assignments, local storage preflight and node profiles |
| Agents | Fixed review instructions and a provider-independent interface; model integration pending |

Autofluorescence and labelled fluorescence can share the numerical tools while
requiring different assay interpretation. Record `--modality` explicitly. Tissue
brightness and empty regions do not independently establish an artifact. A zero
candidate count does not establish acceptable quality.

### Local checks of unflagged regions

```bash
fluoinspect scan /data/export.tif --output /results/local_scan \
  --modality autofluorescence --periodic-screen-state unflagged
```

The scan evaluates every scheduled overlapping tile without a foreground gate.
It proposes dark regions by centre-versus-surround contrast and horizontal or
vertical steps by coherent flank differences plus a narrow boundary check.
Proposals are checked in original uint16 rectangles. Settings are provisional,
and natural tissue spaces/boundaries can produce the same observations.
`local_scan.json` records source hashes, native coordinates, candidate budgets
and detector-evaluated footprints. `tile_review.csv` includes tiles with zero
candidates; `image_review.csv` opens in Excel. Both retain **unverified** status.
Use `--bbox X0 Y0 X1 Y1` for an explicitly supplied, unreviewed analysis envelope,
or `--max-tiles` to bound the scan. Defaults are 1024-pixel tiles, 768-pixel stride
and at most 512 tiles. The independent scan budget does not consume view exports.
Mean-pooled proposal coverage is distinct from exhaustive native-detail or
human/model inspection. Missing tiles, unresolved scales and omitted proposals
are recorded. No outcome assigns acceptance or a cause such as stitching or
background-correction damage. The periodic state is optional caller-supplied
evidence, not a threshold estimated by this command.

`measure --local-artifacts` embeds the same scan in a source-bound measurement
record for evidence packets. Expert annotations are joined only after inference
in private development studies; they are never detector inputs.

`measure` also records experimental brightness-profile measurements. Broad
left/right differences propose context strips in source coordinates. Repetition
requires an explicit native-pixel period, supplied with `--pattern-period-px`;
there is no default acquisition-grid spacing. See the measurement example below.
These are numerical observations, not GRID/stitching classifications or final QC.
Brightness-region selection defaults to a provisional nonzero-intensity Otsu gate;
the edge screen's permissive hysteresis gate can be requested explicitly with
`--pattern-foreground-method hysteresis`. Neither identifies a reviewed biological core.

Linux and Python 3.11+ are currently supported. Multichannel OME-TIFF, image
pyramids and broader WSI formats require additional reader adapters. GX10/Arm64
execution and GPU model serving require target-hardware validation.

## Install and try a synthetic example

```bash
git clone https://github.com/cc111mp/fluoinspect.git
cd fluoinspect
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python examples/make_synthetic.py --output data/inputs/synthetic_step.tif
fluoinspect measure data/inputs/synthetic_step.tif \
  --modality autofluorescence --output work/measurements

fluoinspect inspect create --source data/inputs/synthetic_step.tif \
  --modality autofluorescence --output work/session
fluoinspect inspect view --session work/session --bbox 300 100 500 700 \
  --kind detail --purpose "Inspect synthetic intensity boundary" --request-id boundary-001
fluoinspect packet --session work/session --output work/packet
```

The example contains a known synthetic intensity boundary. It is not a biological
sample or a validation of stitching classification. The packet command above has
no cached measurement input, so its measurements remain explicitly absent. To
bind an existing record, supply `--record` and its independently retained expected
SHA-256 with `--expected-record-sha256`.

Runtime images, model credentials and outputs belong outside Git history. The
repository ships synthetic generators and tests; no study images or run records
are included.

For an independently justified period hypothesis, request repetition measurements:

```bash
fluoinspect measure data/inputs/example.tif --modality autofluorescence \
  --pattern-period-px 128 --pattern-period-basis hypothesis \
  --output work/brightness-measurements
```

Here `128` illustrates a native-pixel period, not a scanner default. Use acquisition
metadata or a documented hypothesis appropriate to the input. Without a period,
periodicity remains unassessed while broad profile measurements are still available.
Poor region coverage, too few supported cycles or inadequate sampling likewise
remain unassessed. Within-image cycle prediction and off-period comparisons are
development evidence; they do not establish biological artifact accuracy.

For brighter–darker–brighter bands, opt in to a separate measurement:

```bash
fluoinspect measure data/inputs/example.tif --modality autofluorescence \
  --alternating-bands --band-region 100,100,1500,1500 \
  --band-widths-px 4,16,64,256 --band-angles-deg 0,90 \
  --output work/band-measurements
```

The rectangle is illustrative and must lie within the source image; it does not
establish a reviewed core. This tool retains dark pixels, proposes regions at
recorded orientations and scales, fits their direction, and checks brighter flanks
in sampled original pixels. It also records low-signal edge straightness, width
variation, exact-zero fractions and relative texture. Two parallel bands with a
brighter interval are an alternating intensity pattern; no acquisition period is
required. These descriptors cannot establish stitching/correction cause or quality
rejection. Ordinary tissue channels can produce similar patterns.

Fine unresolved widths, insufficient flank context and omitted proposals remain
explicit. Inspect the overview and source-bound contexts, including unflagged
regions, before interpreting a negative result. Band measurements are disabled by
default; legacy records retain explicit absence.

## Architecture

Prepare one analysis asset for all measurement tools when a target rectangle has
been selected. This retains original values and keeps child coordinates separate
from parent-export coordinates:

```bash
fluoinspect region --source data/inputs/example.tif --bbox 100 100 1500 1500 \
  --output work/analysis-region
```

Retain the printed receipt SHA-256 independently. Bind the mapping during measurement:

```bash
fluoinspect measure work/analysis-region/region.tif \
  --region-receipt work/analysis-region/region.json \
  --expected-region-receipt-sha256 RECEIPT_SHA256 \
  --pattern-foreground-method region_envelope --alternating-bands \
  --output work/region-measurements
```

The example coordinates must fit the input. A rectangle does not identify a
biological core or exclude neighbours reliably. `region_envelope` measures all
pixels in that analysis image, including internal dim/zero pixels, without
brightness selection or erosion; it may include glass and natural tissue gaps.
It is opt-in and changes the recorded measurement configuration. Existing
nonzero-intensity selection remains the default. Changes in masks, crop support
or coordinate origin require fresh development evaluation and calibration;
do not reuse old thresholds as validated decisions.
For ambiguous or incomplete exports, record `--target-identity-status unresolved`
when preparing the region. This status remains in the bound evidence; it does
not reconstruct missing tissue or establish an intended core.

Create a session and plan or export systematic detail views:

```bash
fluoinspect inspect create --source work/analysis-region/region.tif \
  --output work/region-session
fluoinspect coverage --session work/region-session --tile-size 1024 --stride 768 \
  --export-views --output work/region-coverage
```

The planner respects the remaining session budget and records omitted grid
regions. `--export-views` is optional; a plan alone exports no views. Evidence
packets report the exact union of recorded native crops and full-resolution
previews, so overlapping tiles do not inflate coverage. Exported area is separate
from actual inspection: model/human inspection and automatic acceptance remain
unestablished, even when exported area reaches 100%. All coordinates refer to
the session's input TIFF; the region receipt maps them to its parent export.

```text
src/fluoinspect/
  io/              native reading and durable receipts
  detectors/       artifact candidate measurements
  measurements/    descriptive intensity and detail statistics
  investigation/   sessions, crop navigation and review records
  tools/           method registry and verified evidence packets
  agents/          role instructions and future backend contracts
  deployment/      local preflight and node assignments
  pipeline.py      portable measurement workflow
  cli.py           public command line interface
```

Detectors remain usable without an LLM. The agent layer should request evidence,
report localized observations and preserve uncertainty. Original acquisition
overlaps, before/after corrections and calibration references are needed for
appropriate causal verification. See [architecture](docs/architecture.md),
[validation](docs/validation.md) and [roadmap](docs/roadmap.md).

## Development and contributions

The baseline preparation check passed 114 tests on Linux, including installed-wheel
and synthetic workflows for both modalities. Brightness-pattern challenges extend
this suite to 150 tests; alternating-band geometry and evidence challenges extend
it to 182 tests. Region lineage and coverage challenges extend it to 201 tests.
The CI workflow checks lint, tests and distribution builds
on Python 3.11, 3.12 and 3.13. These checks establish software behavior, not
assay-specific artifact-detection accuracy.

```bash
ruff check src tests examples
pytest
python -m build
```

Contributions are welcome for reader adapters, validated detector methods,
synthetic challenge cases, review interfaces and local model backends. Document
each method's supported scope and failure conditions. See [contributing](CONTRIBUTING.md).

## References and compatibility

The architecture draws on tool-assisted pathology investigation and evidence
verification ideas, including [PathAgent](https://arxiv.org/abs/2511.17052) and
[HarnessIR](https://arxiv.org/abs/2610.10133). These references do not validate
FluoInspect on fluorescence data. [Reference notes](docs/references.md) distinguish
architectural inspiration from implemented methods.

Some versioned JSON schemas retain the `af-qc` namespace for compatibility with
earlier development tools. The public Python package and commands use
`fluoinspect`. Code-origin fingerprints are in [module origins](provenance/module-origins.json).

FluoInspect is distributed under the [MIT license](LICENSE).
