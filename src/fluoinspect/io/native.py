"""Layout-aware, bounded uint16 TIFF reading without source modification."""
from pathlib import Path
import numpy as np
import tifffile

MAX_NATIVE_PIXELS = 256 * 1024 * 1024
MAX_NATIVE_BYTES = 512 * 1024 * 1024

def read_native(path, *, max_pixels=MAX_NATIVE_PIXELS, max_bytes=MAX_NATIVE_BYTES):
    """Decode one uint16 YX plane while preserving pixel values and bit depth.

    Tiled and stripped TIFF segments require their layout-aware decoder. In
    particular, concatenating tile payloads produces a scrambled image even when
    the payload length happens to equal the image length. Missing, empty or
    out-of-file segments are errors, never synthetic zero-valued background.
    Limits apply before decoding to the native array; decoder work buffers and
    downstream feature arrays can require additional process memory.
    """
    if max_pixels <= 0 or max_bytes <= 0:
        raise ValueError("native pixel and byte limits must be positive")
    with tifffile.TiffFile(Path(path)) as tf:
        if len(tf.pages) != 1:
            raise ValueError("expected exactly one TIFF page")
        page = tf.pages[0]
        orientation = page.tags.get("Orientation")
        orientation = int(orientation.value) if orientation is not None else 1
        dtype = np.dtype(page.dtype)
        if (
            len(page.shape) != 2
            or page.axes != "YX"
            or page.samplesperpixel != 1
            or dtype.kind != "u"
            or dtype.itemsize != 2
            or page.bitspersample != 16
            or int(page.photometric) != 1
            or orientation != 1
        ):
            raise ValueError(
                "expected one top-left, minisblack, single-plane uint16 YX page"
            )
        shape = tuple(int(v) for v in page.shape)
        pixels = shape[0] * shape[1]
        if not all(v > 0 for v in shape):
            raise ValueError("empty native image")
        if pixels > max_pixels or pixels * 2 > max_bytes:
            raise ValueError("native decoded image exceeds configured memory limit")
        offsets = tuple(int(v) for v in page.dataoffsets)
        counts = tuple(int(v) for v in page.databytecounts)
        if page.is_tiled:
            expected = (
                (shape[0] + int(page.tilelength) - 1) // int(page.tilelength)
            ) * ((shape[1] + int(page.tilewidth) - 1) // int(page.tilewidth))
        else:
            rows = int(page.rowsperstrip)
            if rows <= 0:
                raise ValueError("invalid TIFF strip height")
            expected = (shape[0] + rows - 1) // rows
        if len(offsets) != expected or len(counts) != expected:
            raise ValueError("TIFF segment count does not cover the native plane")
        file_bytes = int(tf.filehandle.size)
        if any(
            off <= 0 or count <= 0 or off + count > file_bytes
            for off, count in zip(offsets, counts)
        ):
            raise ValueError("TIFF has a missing, empty or out-of-file segment")
        # Decode into ordinary memory from the staged file, never a CIFS memory map.
        native = page.asarray(maxworkers=1)
    if native.shape != shape or native.dtype.kind != "u" or native.dtype.itemsize != 2:
        raise ValueError("decoded TIFF plane disagrees with its native metadata")
    return np.ascontiguousarray(native, dtype=np.uint16)
