# Input provenance

An input TIFF is the source supplied to FluoInspect. Its values may already reflect
background correction, illumination correction, stitching, cropping or export.
Exact pixel preservation establishes that FluoInspect retained those supplied
values; it does not establish that they are unprocessed camera measurements.
A folder named `raw` or `stitch_result`, or a TIFF writer's software tag, cannot
identify the complete acquisition pipeline.

Record the available input stage with the dataset:

| Evidence available | What it supports |
|---|---|
| Original camera fields and acquisition metadata | Acquisition geometry, channel and native sensor units when documented |
| Flat-field/dark-field references and correction settings | Reconstruction and evaluation of the recorded intensity correction |
| Field positions, overlaps and stitcher configuration | Reconstruction and geometric checks of the recorded stitching process |
| Matched images before/after processing | Assessment of changes introduced or left by the documented step |
| Export TIFF only | Measurements and review of the supplied export; upstream cause remains uncertain |

AF tissue brightness is intended signal. Empty background, bright spots, dark
holes and repeated tile patterns are observations requiring context; they do
not independently identify a particular correction algorithm or stitching cause.
TIFF storage strips/tiles are file organization and need not be camera fields.
An analysis grid supplied as a hypothesis stays a hypothesis until acquisition
metadata supports its spacing and origin.

Keep native source bytes and provenance records alongside private dataset
manifests. Public contributions use synthetic examples and documented methods.
FluoInspect currently measures and prepares review evidence; it does not run an
upstream background-correction or stitching pipeline.
