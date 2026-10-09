"""Correctness tests for decoding and positional pattern verification."""

import hashlib

import numpy as np
import pytest
import tifffile


from fluoinspect.detectors import axial as engine


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "shape, kwargs",
    [
        ((32, 48), {"rowsperstrip": 7}),
        ((32, 32), {"tile": (16, 16)}),
        ((35, 50), {"tile": (16, 16)}),
        ((19, 37), {"rowsperstrip": 3, "byteorder": ">"}),
        ((32, 48), {"rowsperstrip": 7, "compression": "deflate"}),
    ],
)
def test_reader_preserves_layout_pixels_and_source(tmp_path, shape, kwargs):
    # Nonuniform values reveal tile reordering, endian changes and edge padding.
    expected = (np.arange(np.prod(shape), dtype=np.uint16) * 17).reshape(shape)
    path = tmp_path / "layout.tif"
    tifffile.imwrite(path, expected, photometric="minisblack", **kwargs)
    before = hash_file(path)
    observed = engine.read_native(path)
    np.testing.assert_array_equal(observed, expected)
    assert observed.dtype == np.dtype(np.uint16)
    assert observed.flags.c_contiguous
    assert hash_file(path) == before


def test_native_memory_limits_are_enforced_before_decoder(tmp_path, monkeypatch):
    path = tmp_path / "native.tif"
    tifffile.imwrite(path, np.ones((32, 48), np.uint16), photometric="minisblack")

    def must_not_decode(*args, **kwargs):
        raise AssertionError("decoder ran before native memory limit was checked")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", must_not_decode)
    with pytest.raises(ValueError, match="memory limit"):
        engine.read_native(path, max_pixels=100)
    with pytest.raises(ValueError, match="memory limit"):
        engine.read_native(path, max_bytes=100)


@pytest.mark.parametrize(
    "array, kwargs",
    [
        (np.zeros((16, 16), np.uint8), {}),
        (np.zeros((16, 16), np.uint16), {"photometric": "miniswhite"}),
        (
            np.zeros((16, 16), np.uint16),
            {"extratags": [(274, "H", 1, 3, False)]},
        ),
        (np.zeros((16, 16, 3), np.uint16), {"photometric": "rgb"}),
        (np.zeros((2, 16, 16), np.uint16), {"photometric": "minisblack"}),
    ],
)
def test_reader_rejects_unsupported_semantics(tmp_path, array, kwargs):
    path = tmp_path / "unsupported.tif"
    tifffile.imwrite(path, array, **kwargs)
    with pytest.raises(ValueError, match="expected"):
        engine.read_native(path)


def test_reader_rejects_missing_segment_before_decoder(tmp_path, monkeypatch):
    path = tmp_path / "missing.tif"
    tifffile.imwrite(path, np.ones((16, 16), np.uint16), rowsperstrip=4)
    with tifffile.TiffFile(path) as tf:
        tag = tf.pages[0].tags["StripByteCounts"]
        position = tag.valueoffset
        width = {3: 2, 4: 4, 16: 8}[int(tag.dtype)]
    data = bytearray(path.read_bytes())
    data[position : position + width] = bytes(width)
    path.write_bytes(data)

    def must_not_decode(*args, **kwargs):
        raise AssertionError("decoder ran on an empty TIFF segment")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", must_not_decode)
    with pytest.raises(ValueError, match="missing, empty or out-of-file"):
        engine.read_native(path)


def test_reader_rejects_truncated_segment(tmp_path):
    path = tmp_path / "truncated.tif"
    tifffile.imwrite(path, np.ones((32, 48), np.uint16), rowsperstrip=7)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="out-of-file"):
        engine.read_native(path)


def test_overview_and_pixel_hash_preserve_remainder_pixels(tmp_path):
    expected = np.arange(35 * 50, dtype=np.uint16).reshape(35, 50)
    path = tmp_path / "overview.tif"
    tifffile.imwrite(path, expected, tile=(16, 16))
    native, overview, pixel_hash = engine.load_overview(path)
    np.testing.assert_array_equal(native, expected)
    np.testing.assert_allclose(
        overview,
        expected[:32, :48].astype(np.float32).reshape(4, 8, 6, 8).mean((1, 3)),
    )
    assert pixel_hash == hashlib.sha256(expected.tobytes()).hexdigest()


@pytest.mark.parametrize("vertical", [True, False])
def test_wide_means_verify_same_position_and_ignore_outer_stripe(vertical):
    native = np.full((400, 200), 100, np.uint16)
    # Candidate at x=96 searches [56,136]. The thin stripe at62 lies outside
    # complete16px side support; old code verified it at a different position.
    native[:, 62] = 600
    native[:, 76:] += 100
    if not vertical:
        native = native.T.copy()
    candidate = dict(pos=12, start=0, end=50)
    result = engine.verify(native, candidate, vertical, low_raw=150, contrast=1000)
    assert result["exact_rows"] == 400
    assert 72 <= result["native_line"] <= 76
    assert result["median_jump"] == pytest.approx(0.1)


def test_outer_thin_stripe_alone_is_unverified():
    native = np.full((400, 200), 100, np.uint16)
    native[:, 62] = 600
    result = engine.verify(
        native, dict(pos=12, start=0, end=50), True, low_raw=150, contrast=1000
    )
    assert result == dict(exact_rows=0, native_line=None, median_jump=0.0)


@pytest.mark.parametrize("vertical", [True, False])
def test_real_step_retained_and_too_narrow_strip_unverified(vertical):
    native = np.full((400, 200), 100, np.uint16)
    native[:, 96:] = 1000
    if not vertical:
        native = native.T.copy()
    result = engine.verify(
        native, dict(pos=12, start=0, end=50), vertical, low_raw=200, contrast=1000
    )
    assert result["exact_rows"] == 400
    assert 93 <= result["native_line"] <= 96
    assert result["median_jump"] == pytest.approx(0.9)
    narrow = np.ones((400, 20), np.uint16) * 100
    if not vertical:
        narrow = narrow.T.copy()
    result = engine.verify(
        narrow, dict(pos=1, start=0, end=50), vertical, low_raw=200, contrast=1000
    )
    assert result["exact_rows"] == 0


def test_pattern_verification_does_not_claim_artifact_cause(tmp_path):
    native = np.full((800, 800), 100, np.uint16)
    native[:, 400:] = 1000
    path = tmp_path / "step.tif"
    tifffile.imwrite(path, native, rowsperstrip=80)
    _, _, lines, _, shape = engine.screen(path)
    assert shape == native.shape
    assert any(line["verified_pattern"] for line in lines)
    assert all(line["confirmed"] == line["verified_pattern"] for line in lines)
    assert all("artifact_class" not in line for line in lines)


def test_screen_native_matches_file_api_and_preserves_input(tmp_path):
    native = np.full((800, 800), 100, np.uint16)
    native[:, 400:] = 1000
    path = tmp_path / "step.tif"
    tifffile.imwrite(path, native, tile=(16, 16))
    before = native.copy()
    native.flags.writeable = False
    memory_result = engine.screen_native(native)
    file_result = engine.screen(path)
    np.testing.assert_array_equal(native, before)
    np.testing.assert_array_equal(memory_result[0], file_result[0])
    np.testing.assert_array_equal(memory_result[1], file_result[1])
    assert memory_result[2:] == file_result[2:]


def test_ellipse_is_a_location_rule_and_legacy_alias_matches():
    line = dict(orientation="vertical", pos=500, start=300, end=700)
    assert engine.in_central_ellipse(line, (8000, 8000))
    assert engine.in_core(line, (8000, 8000))
    border_line = dict(line, pos=10)
    assert not engine.in_central_ellipse(border_line, (8000, 8000))
    assert not engine.in_core(border_line, (8000, 8000))


def test_overlay_write_failure_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(engine.cv2, "imwrite", lambda *args, **kwargs: False)
    with pytest.raises(OSError, match="could not write"):
        engine.draw(np.ones((100, 100), np.float32), [], tmp_path / "failed.png")


def test_overlay_is_written_at_verified_native_position(tmp_path):
    path = tmp_path / "overlay.png"
    view = np.ones((128, 128), np.float32)
    line = dict(
        verified_pattern=True,
        orientation="vertical",
        pos=20,
        native_line=400,
        start=30,
        end=100,
        in_central_ellipse=True,
    )
    engine.draw(view, [line], path)
    image = engine.cv2.imread(str(path))
    assert image is not None
    assert image.shape == (1024, 1024, 3)
    # Native400/8 = overview50; display scales by8, givingx400.
    assert image[500, 400].tolist() == [0, 0, 255]
    assert image[500, 160].tolist() == [255, 255, 255]


@pytest.mark.parametrize("transpose", [False, True])
def test_one_native_step_has_one_scoring_representative(transpose):
    yy, xx = np.indices((1024, 1280))
    native = (100 + yy // 4 + xx // 8 + (xx >= 640) * 1500).astype(np.uint16)
    if transpose:
        native = native.T.copy()
    _, _, lines, _, _ = engine.screen_native(native)
    verified = [line for line in lines if line["verified_pattern"]]
    assert len(verified) == 2  # Both original overview proposals remain inspectable.
    assert {line["native_line"] for line in verified} == {640}
    unique = [line for line in lines if line["unique_verified_pattern"]]
    assert len(unique) == 1
    assert unique[0]["exact_rows"] == 1024
    assert unique[0]["duplicate_of_candidate_index"] is None
    duplicate = next(line for line in verified if not line["unique_verified_pattern"])
    assert lines[duplicate["duplicate_of_candidate_index"]] is unique[0]


def test_native_deduplication_retains_strength_and_distinct_evidence():
    def candidate(position, start, end, rows, orientation="vertical"):
        return dict(
            pos=position // engine.F,
            start=start,
            end=end,
            exact_rows=rows,
            median_jump=0.2,
            native_line=position,
            orientation=orientation,
            source="step",
            verified_pattern=rows >= engine.CONFIRM_ROWS,
        )

    lines = [
        candidate(640, 0, 80, 400),
        candidate(641, 10, 90, 600),  # Stronger, overlapping, within one nativepx.
        candidate(640, 100, 160, 450),  # Separate span must remain separate.
        candidate(640, 0, 80, 500, "horizontal"),  # Crossing orientation remains.
        candidate(645, 0, 80, 500),  # Another native axis remains.
        candidate(640, 0, 80, 200),  # Unverified candidate remains inspectable.
    ]
    before_rows = [line["exact_rows"] for line in lines]
    engine._mark_unique_verified_patterns(lines)
    assert [line["exact_rows"] for line in lines] == before_rows
    assert [line["unique_verified_pattern"] for line in lines] == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert lines[0]["duplicate_of_candidate_index"] == 1
    assert lines[0]["verified_pattern"]
    assert lines[5]["duplicate_of_candidate_index"] is None


def test_native_deduplication_ties_are_deterministic():
    a = dict(
        pos=78,
        start=0,
        end=128,
        exact_rows=1024,
        median_jump=0.2,
        native_line=640,
        orientation="vertical",
        source="step",
        verified_pattern=True,
    )
    b = dict(a, pos=82)
    for lines in ([dict(a), dict(b)], [dict(b), dict(a)]):
        engine._mark_unique_verified_patterns(lines)
        retained = next(line for line in lines if line["unique_verified_pattern"])
        assert retained["pos"] == 78
