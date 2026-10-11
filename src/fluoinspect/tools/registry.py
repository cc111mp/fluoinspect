"""Method availability and interpretation limits; no inferred quality acceptance."""
TOOL_SCOPES = {
    "source_and_crop_integrity": {"implemented": True, "scope": "file and pixel integrity; no quality label"},
    "context_and_native_views": {"implemented": True, "scope": "recorded source rectangles and resolution"},
    "analysis_region_lineage": {"implemented": True, "scope": "exact crop/parent coordinate binding; supplied core envelope remains unreviewed"},
    "systematic_detail_coverage": {"implemented": True, "scope": "bounded grid planning and exported-area union; does not establish model/human inspection"},
    "experimental_local_artifacts": {"implemented": True, "scope": "systematic tiles including unflagged areas; local dark-region and axial-step proposals checked in original pixels; cause/accuracy unvalidated"},
    "review_triage": {"implemented": True, "scope": "code-owned review routing; negative/unassessed screens remain unverified and no profile acceptance is assigned"},
    "experimental_core_regions": {"implemented": True, "scope": "coarse traditional core proposals and separate tissue/background labels; retain internal dark values; core identity/boundary accuracy unvalidated"},
    "core_candidate_attribution": {"implemented": True, "scope": "attribute unchanged native measurements to provisional regions; no source masking, defect localization gold or QC acceptance"},
    "sharp_axial_pattern_screen": {"implemented": True, "scope": "sharp horizontal/vertical candidates; AF accuracy unvalidated"},
    "supporting_intensity_background_detail": {"implemented": True, "scope": "measurements only; masks and quality interpretation unvalidated"},
    "experimental_brightness_patterns": {"implemented": True, "scope": "broad axis-profile variation and explicit-period repetition evidence; regions and artifact accuracy unvalidated"},
    "experimental_alternating_bands": {"implemented": True, "scope": "opt-in dark-band and parallel bright/dark geometry; recorded scales/orientations, sampled original-pixel verification; cause and quality interpretation unvalidated"},
    "broad_oblique_band_detector": {"implemented": False, "scope": "requires detector development and validation"},
    "geometric_overlap_verifier": {"implemented": False, "scope": "requires original acquisition fields/reference evidence"},
    "correction_damage_verifier": {"implemented": False, "scope": "requires before/after or calibration evidence"},
}
