"""Bound numerical threads for predictable CPU tests."""
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import cv2  # noqa: E402

cv2.setNumThreads(1)
