"""Native percentile, clipping, shared transfer and saved-session compatibility."""
import json

import numpy as np
from PIL import Image
import pytest
import tifffile

from fluoinspect.io import display
from fluoinspect.investigation import session as zoom


def test_histogram_percentiles_match_direct_native_population():
    values = np.array([0, 0, 15, 20, 200, 200, 1000, 65535], dtype=np.uint16).reshape(2, 4)
    counts = display.histogram(values)
    assert int(counts.sum()) == values.size
    for q in [0, 1, 50, 90, 99.9, 100]:
        assert display.percentile(counts, q) == pytest.approx(np.percentile(values, q))


def test_known_bright_crop_protects_shared_window_from_dark_population():
    values = np.zeros((1000, 1000), dtype=np.uint16)
    values[0, :100] = np.arange(10000, 10100, dtype=np.uint16)
    region_counts = [display.histogram(values), display.histogram(values[:1, :100])]
    assert display.shared_window(region_counts) == [0, 10099]
    assert all(display.clipping(h, [0, 10099])["above_percent"] == 0 for h in region_counts)


def test_equal_values_keep_equal_gray_across_views_and_source_is_preserved():
    values = np.array([[0, 1000, 2000, 4000, 8000]], dtype=np.uint16)
    original = values.tobytes()
    values.flags.writeable = False
    assert display.render(values, [0, 8000]).tolist() == [[0, 32, 64, 128, 255]]
    for mode in ["linear", "log"]:
        np.testing.assert_array_equal(display.render(values, [0, 8000], mode)[:, 1:4],
                                      display.render(values[:, 1:4], [0, 8000], mode))
    assert values.tobytes() == original


def test_clipping_uses_original_values_before_png_rounding():
    values = np.array([[0, 1, 100, 200, 201]], dtype=np.uint16)
    assert display.clipping(display.histogram(values), [1, 200]) == {
        "below_percent": 20., "above_percent": 20.}
    # Rounding maps an in-window value to white; that value is not clipped.
    near_white = np.array([[999, 1000, 1001]], dtype=np.uint16)
    assert display.render(near_white, [0, 1000]).tolist() == [[255, 255, 255]]
    assert display.clipping(display.histogram(near_white), [0, 1000])["above_percent"] == pytest.approx(100 / 3)


def test_zero_image_has_valid_window_and_log_is_explicit():
    values = np.zeros((20, 20), dtype=np.uint16)
    assert display.shared_window([display.histogram(values)]) == [0, 1]
    assert not display.render(values, [0, 1]).any()
    dim = np.array([[100]], dtype=np.uint16)
    assert display.render(dim, [0, 10000], "linear")[0, 0] == 3
    assert display.render(dim, [0, 10000], "log")[0, 0] > 100


@pytest.mark.parametrize("window", [[100, 100], [-1, 50], [0, 65536], [0, float("nan")], [0]])
def test_invalid_display_windows_are_rejected(window):
    with pytest.raises(ValueError):
        display.render(np.ones((1, 1), dtype=np.uint16), window)


@pytest.mark.parametrize("q", [-1, 101, float("nan")])
def test_invalid_percentiles_are_rejected(q):
    with pytest.raises(ValueError):
        display.percentile(display.histogram(np.zeros((1, 1), dtype=np.uint16)), q)


@pytest.mark.parametrize("values", [np.empty((0, 2), dtype=np.uint16),
                                  np.ones((2, 2), dtype=np.float32),
                                  np.ones(2, dtype=np.uint16)])
def test_histogram_requires_valid_native_plane(values):
    with pytest.raises(ValueError):
        display.histogram(values)


@pytest.mark.parametrize('operation', ['histogram', 'render', 'percentile'])
def test_masked_images_and_histograms_are_rejected(operation):
    pixels = np.ma.array(np.full((8, 8), 100, np.uint16), mask=True)
    with pytest.raises(ValueError, match='Masked'):
        if operation == 'histogram':
            display.histogram(pixels)
        elif operation == 'render':
            display.render(pixels, [0, 200])
        else:
            counts = np.ma.array(display.histogram(pixels.data), mask=True)
            display.percentile(counts, 50)


def test_renderer_accepts_native_unit_block_means_and_rejects_invalid_signal():
    means = np.array([[0., 4000.]], dtype=np.float32)
    assert display.render(means, [0, 8000]).tolist() == [[0, 128]]
    for invalid in [float("nan"), float("inf"), -1., 65536.]:
        with pytest.raises(ValueError):
            display.render(np.array([[invalid]]), [0, 8000])
    with pytest.raises(ValueError):
        display.render(means, [0, 8000], "auto")
    with pytest.raises(ValueError):
        display.shared_window([])
    with pytest.raises(ValueError):
        display.percentile(np.zeros(65536, dtype=np.uint64), 50)


def _source(tmp_path, values):
    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "synthetic.tif"
    tifffile.imwrite(source, values, photometric="minisblack")
    return source


def test_session_defaults_use_native_population_and_fixed_linear_window(tmp_path):
    values = np.zeros((1200, 1200), dtype=np.uint16)
    values[::4, ::4] = 16000
    source = _source(tmp_path, values)
    before = zoom.sha(source)
    session = tmp_path / "session"
    overview = zoom.create_session(source, session)
    # Bright native values remain in the percentile despite overview averaging.
    assert overview["display_window"]["high"] == 16000
    assert overview["display_window"]["mode"] == "linear"
    assert overview["display_window"]["method"] == "native_p99.9"
    detail = zoom.request_view(session, [0, 0, 20, 20], purpose="Check synthetic native values")
    assert detail["display_window"] == overview["display_window"]
    assert detail["display_clipping_native_pixels"]["above_percent"] == 0
    np.testing.assert_array_equal(np.asarray(Image.open(zoom.artifact_path(session, detail["png"]))),
                                  display.render(values[:20, :20], [0, 16000]))
    np.testing.assert_array_equal(tifffile.imread(zoom.artifact_path(session, detail["raw_tiff"])),
                                  values[:20, :20])
    assert zoom.sha(source) == before


def test_later_bright_crop_reports_clipping_without_changing_prior_views(tmp_path):
    values = np.zeros((1000, 1000), dtype=np.uint16)
    values[:10, :10] = 10000
    source = _source(tmp_path, values)
    session = tmp_path / "session"
    overview = zoom.create_session(source, session)
    previous_hash = overview["png_sha256"]
    detail = zoom.request_view(session, [0, 0, 10, 10], purpose="Inspect synthetic bright spot")
    assert overview["display_window"]["high"] == 1
    assert detail["display_window"] == overview["display_window"]
    assert detail["display_clipping_native_pixels"]["above_percent"] == 100
    assert zoom.sha(zoom.artifact_path(session, overview["png"])) == previous_hash


def test_log_session_is_explicit_and_native_crop_remains_exact(tmp_path):
    values = np.full((20, 20), 100, dtype=np.uint16)
    source = _source(tmp_path, values)
    session = tmp_path / "session"
    overview = zoom.create_session(source, session, display_high=10000, display_mode="log")
    detail = zoom.request_view(session, [0, 0, 10, 10], purpose="Inspect low native signal")
    assert detail["display_window"] == overview["display_window"]
    assert detail["display_window"]["mode"] == "log"
    assert np.asarray(Image.open(zoom.artifact_path(session, detail["png"])))[0, 0] > 100
    np.testing.assert_array_equal(tifffile.imread(zoom.artifact_path(session, detail["raw_tiff"])), values[:10, :10])


@pytest.mark.parametrize("high", [60000, 70000])
def test_saved_legacy_gamma_sessions_preserve_existing_assets_and_mapping(tmp_path, high):
    values = np.arange(400, dtype=np.uint16).reshape(20, 20) * 100
    source = _source(tmp_path, values)
    session = tmp_path / "session"
    overview = zoom.create_session(source, session, display_high=60000)
    info = json.loads((session / "session.json").read_text())
    legacy_window = {"low": 0, "high": high, "gamma_power": .5}
    info["display_window"] = legacy_window
    info["engine_revision"] = "zoom-tools-robust-v2"
    info["views"][0]["display_window"] = legacy_window
    info["views"][0].pop("display_clipping_native_pixels")
    original_png = zoom.artifact_path(session, overview["png"])
    legacy_pixels = (np.sqrt(values.astype(np.float32) / high) * 255).astype(np.uint8)
    Image.fromarray(legacy_pixels).save(original_png)
    old_hash = zoom.sha(original_png)
    info["views"][0]["png_sha256"] = old_hash
    zoom.atomic_json(session / "session.json", info)
    detail = zoom.request_view(session, [0, 0, 10, 10], purpose="Resume a legacy synthetic session")
    assert detail["display_window"] == legacy_window
    np.testing.assert_array_equal(np.asarray(Image.open(zoom.artifact_path(session, detail["png"]))),
                                  legacy_pixels[:10, :10])
    assert zoom.sha(original_png) == old_hash
