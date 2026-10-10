"""Dark retention, neighbour exclusion, ambiguity and source-bound scope attribution."""
import copy
import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from fluoinspect.detectors.local import LocalConfig, measure_local_artifacts
from fluoinspect.investigation.local_scan import scan_image, write_scan
from fluoinspect.investigation.session import sha
from fluoinspect.segmentation.analysis import attribute_candidates
from fluoinspect.segmentation.core import CoreConfig, label_counts_in_box, measure_region_intensity, propose_core
from fluoinspect.segmentation.workflow import analyze_core, verify_core_result, write_core_result


def fixture(shape=(513, 611), gap=False):
    y, x = np.indices(shape)
    pixels = np.full(shape, 50, np.uint16)
    center = [shape[1] // 2, shape[0] // 2]
    core = (x - center[0]) ** 2 + (y - center[1]) ** 2 < 130 ** 2
    pixels[core] = 1500
    # A partial neighbouring core touches the right export boundary.
    neighbour = (x - (shape[1] + 45)) ** 2 + (y - 160) ** 2 < 130 ** 2
    pixels[neighbour] = 2400
    pixels[230:260, 285:320] = 0
    if gap:
        pixels[core & (np.abs(x - center[0]) < 14)] = 0
    return pixels, core, neighbour


def config(**options):
    return CoreConfig(max_overview_dimension=256, **options)


def test_envelope_retains_internal_zeros_and_support_stays_separate():
    pixels, truth, neighbour = fixture()
    before = pixels.copy()
    result, labels = propose_core(pixels, config=config())
    assert result['target_identity_status'] == 'unreviewed'
    step = result['native_pixels_per_label_cell']
    center = labels[245 // step, 300 // step]
    assert center == 1  # dark-retaining envelope, not bright tissue support
    assert labels[160 // step, 600 // step] == 4
    measured = measure_region_intensity(pixels, labels, step)
    assert measured['core_envelope']['zero_pixels'] >= 900
    assert measured['tissue_support']['zero_pixels'] < measured['core_envelope']['zero_pixels']
    assert np.array_equal(before, pixels)
    assert result['internal_dark_pixels_retained'] and not result['core_boundary_reviewed']
    assert result['quality_decision'] is None
    assert sum(result['label_areas_native_px'].values()) == pixels.size


def test_a_dark_line_splitting_core_is_retained_in_proposed_envelope():
    pixels, _, _ = fixture(gap=True)
    result, labels = propose_core(pixels, config=config())
    step = result['native_pixels_per_label_cell']
    assert labels[210 // step, (pixels.shape[1] // 2) // step] in (1, 2)
    assert measure_region_intensity(pixels, labels, step)['core_envelope']['zero_pixels'] > 5000
    assert result['source_pixels_modified'] is False


@pytest.mark.parametrize('box', [[0, 0, 11, 9], [2, 1, 10, 8], [8, 6, 11, 9]])
def test_partial_cell_area_mapping_matches_independent_native_raster(box):
    labels = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 0]], np.uint8)
    expanded = labels.repeat(3, axis=0).repeat(3, axis=1)[:9, :11]
    x0, y0, x1, y1 = box
    result = label_counts_in_box(labels, 3, [9, 11], box)
    assert list(result.values()) == [int(np.count_nonzero(expanded[y0:y1, x0:x1] == code)) for code in range(5)]


@pytest.mark.parametrize('kind', ['empty', 'constant', 'two_partial', 'elongated'])
def test_missing_partial_or_elongated_targets_remain_unresolved(kind):
    pixels = np.full((512, 512), 50, np.uint16)
    y, x = np.indices(pixels.shape)
    if kind == 'two_partial':
        pixels[((x - 256) ** 2 + (y - 0) ** 2 < 180 ** 2)
               | ((x - 256) ** 2 + (y - 512) ** 2 < 180 ** 2)] = 1500
    elif kind == 'elongated':
        pixels[50:460, 210:300] = 1500
    elif kind == 'empty':
        pixels[:] = 0
    result, _ = propose_core(pixels, config=config())
    assert result['target_identity_status'] == 'unresolved'
    assert result['unresolved_reasons'] and result['quality_decision'] is None


def test_equal_interior_core_proposals_are_ambiguous_and_point_remains_unreviewed():
    y, x = np.indices((512, 800))
    pixels = np.full((512, 800), 50, np.uint16)
    pixels[((x - 220) ** 2 + (y - 256) ** 2 < 120 ** 2)
           | ((x - 580) ** 2 + (y - 256) ** 2 < 120 ** 2)] = 1500
    ambiguous, _ = propose_core(pixels, config=config())
    assert ambiguous['target_identity_status'] == 'unresolved'
    assert 'multiple_similarly_ranked_core_proposals' in ambiguous['unresolved_reasons']
    explicit, _ = propose_core(pixels, config=config(), target_point_native_xy=[220, 256])
    assert explicit['selection_basis'] == 'caller_supplied_unreviewed_point'
    assert explicit['target_identity_status'] == 'unreviewed' and not explicit['core_boundary_reviewed']


def test_interior_half_core_has_flat_edge_review_flag_without_border_contact():
    y, x = np.indices((512, 512))
    pixels = np.full((512, 512), 50, np.uint16)
    pixels[((x - 256) ** 2 + (y - 256) ** 2 < 150 ** 2) & (y < 256)] = 1500
    result, _ = propose_core(pixels, config=config(max_axis_ratio=3))
    assert result['target_identity_status'] == 'unresolved'
    assert 'selected_polygon_has_long_flat_edge' in result['unresolved_reasons']
    assert 'selected_support_touches_export_border' not in result['unresolved_reasons']


def test_empty_regions_do_not_receive_a_defined_region_coverage_fraction():
    pixels = np.zeros((256, 256), np.uint16)
    proposal, labels = propose_core(pixels, config=config())
    local = measure_local_artifacts(pixels)
    assert local['coverage']['detector_evaluated_area_fraction'] == 1
    attributed = attribute_candidates(local, proposal, labels)
    assert attributed['core_and_background_detector_footprint_fraction'] is None


def test_masked_source_arrays_are_rejected():
    pixels = np.ma.array(fixture()[0], mask=False)
    with pytest.raises(ValueError):
        propose_core(pixels)


def test_region_attribution_retains_same_detector_scores_and_neighbour_candidates():
    pixels, _, _ = fixture()
    proposal, labels = propose_core(pixels, config=config())
    local = measure_local_artifacts(pixels, config=LocalConfig(tile_size=512, stride=384, widths_native_px=(16, 64)))
    before = copy.deepcopy(local)
    attributed = attribute_candidates(local, proposal, labels)
    assert len(attributed['candidates']) == len(local['candidates']) > 0
    assert [row['native_log_contrast'] for row in attributed['candidates']] == [row['native_log_contrast'] for row in local['candidates']]
    assert attributed['counts_by_scope']['core_interior'] > 0
    assert local == before and not attributed['source_was_masked_before_detection']
    assert attributed['core_and_background_detector_footprint_fraction'] == 1
    assert attributed['quality_decision'] is None


@pytest.mark.parametrize('budget', [0, 1])
def test_incomplete_detector_coverage_does_not_imply_complete_region_inspection(budget):
    pixels, _, _ = fixture()
    proposal, labels = propose_core(pixels, config=config())
    local = measure_local_artifacts(pixels, config=LocalConfig(tile_size=256, stride=192, max_tiles=budget))
    attributed = attribute_candidates(local, proposal, labels)
    assert attributed['core_and_background_detector_footprint_fraction'] is None
    assert attributed['model_inspected_area_fraction'] is None and attributed['quality_decision'] is None


@pytest.mark.parametrize('options', [{'max_objects':True}, {'closing_fraction':float('nan')},
                                    {'max_axis_ratio':1}, {'background_ring_fraction':0}, {'max_overview_dimension':4096}])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        CoreConfig(**options)


@pytest.mark.parametrize('point', [[-1, 2], [1, 999], [float('nan'), 1], [1], ['1', 2]])
def test_invalid_target_prior_is_rejected(point):
    with pytest.raises(ValueError):
        propose_core(fixture()[0], target_point_native_xy=point)


def sources(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    source = inputs / 'core.tif'
    tifffile.imwrite(source, fixture()[0], photometric='minisblack')
    return source


def test_cached_local_record_has_independent_hash_and_current_source_binding(tmp_path):
    source = sources(tmp_path)
    cached = scan_image(source)
    cached_dir = tmp_path / 'local'
    write_scan(cached_dir, cached)
    path = cached_dir / 'local_scan.json'
    expected = sha(path)
    with pytest.raises(ValueError, match='retained'):
        analyze_core(source, cached_local_scan=path)
    result, labels = analyze_core(source, cached_local_scan=path, expected_local_scan_sha256=expected)
    assert result['cached_local_lineage']['sha256'] == expected
    assert result['local_candidate_attribution']['all_original_candidates_retained']
    before = sha(source)
    saved = write_core_result(tmp_path / 'regions', result, labels)
    checked, checked_labels = verify_core_result(tmp_path / 'regions/core_regions.json',
                                                expected_record_sha256=sha(tmp_path / 'regions/core_regions.json'))
    assert saved == checked and np.array_equal(labels, checked_labels)
    assert sha(source) == before and checked['review_triage']['verification_state'] == 'unverified'
    assert all(value is None for value in checked['review_triage']['profile_decisions'].values())


@pytest.mark.parametrize('mutation', ['source', 'mask', 'geometry', 'reviewed', 'accepted'])
def test_changed_source_masks_geometry_and_forged_review_claims_are_rejected(tmp_path, mutation):
    source = sources(tmp_path)
    result, labels = analyze_core(source)
    write_core_result(tmp_path / 'regions', result, labels)
    path = tmp_path / 'regions/core_regions.json'
    record = json.loads(path.read_text())
    expected = sha(path)
    if mutation == 'source':
        pixels = tifffile.imread(source)
        pixels[0, 0] += 1
        tifffile.imwrite(source, pixels, photometric='minisblack')
    elif mutation == 'mask':
        labels[40, 40] = 4
        tifffile.imwrite(path.parent / 'regions.tif', labels, photometric='minisblack')
    else:
        if mutation == 'geometry':
            record['proposal']['core_envelope_bbox_level0_xyxy'][0] += 1
        elif mutation == 'reviewed':
            record['proposal']['core_boundary_reviewed'] = True
        else:
            record['review_triage']['quality_decision'] = 'accepted'
        path.write_text(json.dumps(record))
        expected = sha(path)
    with pytest.raises(ValueError):
        verify_core_result(path, expected_record_sha256=expected)


def test_cli_keeps_expert_and_profile_fields_blank(tmp_path):
    source = sources(tmp_path)
    destination = tmp_path / 'regions'
    process = subprocess.run([sys.executable, '-m', 'fluoinspect', 'core', str(source), '--output', str(destination),
                              '--modality', 'fluorescence'], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    record = json.loads((destination / 'core_regions.json').read_text())
    assert record['modality'] == 'fluorescence' and record['quality_decision'] is None
    row = next(csv.DictReader((destination / 'core_review.csv').open(encoding='utf-8-sig')))
    assert row['Verification_state'] == 'unverified' and row['Expert_core_identity_confirmed'] == ''
    assert row['Final_quality_decision'] == ''
