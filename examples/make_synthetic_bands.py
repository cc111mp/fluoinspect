"""Generate known alternating intensity bands; never a biological QC benchmark."""
import argparse
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--angle-deg", type=float, default=0.6)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Synthetic output must be a new file")
    if not np.isfinite(args.angle_deg):
        raise ValueError("Angle must be finite")
    y, x = np.indices((1600, 1600))
    normal = -np.sin(np.deg2rad(args.angle_deg)) * (x - 800) + np.cos(np.deg2rad(args.angle_deg)) * (y - 800)
    dark = (np.abs(normal - 250) <= 8) | (np.abs(normal + 250) <= 8)
    image = np.where(dark, 0, 4000).astype(np.uint16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.output, image, photometric="minisblack")
    print(f"Created known synthetic bright/dark bands at {args.output}")


if __name__ == "__main__":
    main()
