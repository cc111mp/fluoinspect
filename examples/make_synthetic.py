"""Generate an explicitly synthetic intensity-boundary example."""
import argparse
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing examples; use a fresh output path")
    values = np.full((800, 800), 100, np.uint16)
    values[:, 400:] = 1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output, values, photometric="minisblack")


if __name__ == "__main__":
    main()
