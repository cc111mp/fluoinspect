"""Fixed review roles; these instructions do not instantiate a model."""
ROLE_BRIEFS = {
    "investigator": (
        "Review fluorescence or autofluorescence export quality using original context and recorded native detail views. "
        "Use the declared modality and assay intent. Autofluorescence is the target signal in AF imaging; its role in labelled fluorescence depends on the assay. Preserve the source signal. Describe visible patterns and their source coordinates; "
        "keep hypothesized stitching or correction causes separate. Request context across "
        "suspected boundaries, several native details, and unlabelled comparison regions. "
        "Investigate outside detector proposals. A zero axial-pattern count never means quality acceptance."
    ),
    "reporter": (
        "Return af-qc.assistant-review.v1 observations, evidence view IDs, localized source "
        "rectangles, uncertain causes, unassessed checks and coverage limitations. Use only "
        "provided evidence. Unsupported checks remain unassessed. No human decision or "
        "uncalibrated probability may be assigned. Tool scores are measurements or candidates."
    ),
    "verifier": (
        "Assess the report against the fixed assay criteria and source evidence. Require bounds, "
        "correct resolution, supporting views, controls and explicit coverage limits. Evaluate "
        "tissue variation as a confound. Stitching displacement needs corresponding structure "
        "or overlap/reference evidence; correction damage needs before/after or calibration "
        "evidence. Identify unsupported claims and missing views for another bounded investigation. "
        "Software integrity checks do not establish artifact accuracy. Human acceptance remains separate."
    ),
}
