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
| Axial screening | Sharp horizontal/vertical intensity-pattern candidates; not causal stitching labels |
| Investigation | Overview, context, detail and unlabelled comparison views with source coordinates |
| Evidence | Source hashes, exact native crop readback, record binding and explicit unassessed checks |
| Deployment | Fixed job assignments, local storage preflight and node profiles |
| Agents | Fixed review instructions and a provider-independent interface; model integration pending |

Autofluorescence and labelled fluorescence can share the numerical tools while
requiring different assay interpretation. Record `--modality` explicitly. Tissue
brightness and empty regions do not independently establish an artifact. A zero
candidate count does not establish acceptable quality.

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

## Architecture

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
this suite to 150 tests. The CI workflow checks lint, tests and distribution builds
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
