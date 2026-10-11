# Review display

New `inspect create` sessions use a linear transfer from stored intensity to
8-bit gray. The whole-image overview and every native crop share the same fixed
black/white limits. Default black is zero; default white is the ceiling of the
native source's 99.9th percentile, with a minimum of one. A downsampled overview
does not select the window. This is a per-image display policy, not cross-image
intensity calibration.

Every view records `display_window` and `display_clipping_native_pixels`. Clipping
percentages count original pixels strictly below/above the limits, before PNG
rounding. Native TIFF crops retain exact source values. Overview pixels are means
of native blocks; their recorded geometry includes partial edge blocks.

The default source window stays fixed when later crops arrive. A rare bright crop
can have substantial clipping even when whole-image clipping is small. Inspect
the native TIFF or start a fresh session with a justified explicit white limit:

```bash
fluoinspect inspect create --source /data/export.tif --output /results/linear \
  --display-high 20000 --display-mode linear
```

For a fixed panel whose detail crops are already known, the reusable display
functions support the policy used by the private review builder: black=0 and
white equal to the largest native 99.9th percentile among the source and all
displayed crops. This usually limits upper clipping to approximately 0.1% in
those regions; finite population interpolation and ties affect the exact amount.

```python
from fluoinspect.io.display import clipping, histogram, render, shared_window

regions = [native_export, strongest_native_crop, weakest_native_crop]
counts = [histogram(region) for region in regions]
window = shared_window(counts)
previews = [render(region, window) for region in regions]
clipping_percentages = [clipping(h, window) for h in counts]
```

Choose log display explicitly with `--display-mode log`. It uses the same native
window endpoints and can reveal faint background. Display contrast alone does
not establish artifact severity or correction history. Display functions do not
feed detectors, subtract background, fill dark regions or normalize the source.

Saved sessions without a `mode` field keep their historical `gamma_power`
transfer when resumed. Existing previews and evidence hashes are preserved.
New linear/log sessions use round-to-nearest gray; legacy sessions retain their
historical truncation. Display modes and window choices are recorded rather than
silently changed within a session.
