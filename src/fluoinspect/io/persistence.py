"""Standard-library-only persistence and budget contract for zoom tools."""
import json
import os
from pathlib import Path
import tempfile

DEFAULT_BUDGET = {"max_views": 16, "max_display_dimension": 1024,
                  "max_raw_crop_pixels": 4 * 1024 * 1024}


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                         encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
