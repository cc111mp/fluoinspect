# Changelog

## Unreleased

- Consolidate experimental brightness, alternating-band, local artifact and core
  region tools in the development toolkit.
- Use a fixed shared linear window for new investigation previews, derived from
  native intensities rather than downsampled display values.
- Record original-pixel clipping percentages for each view; support explicit log
  display and shared windows protecting known detail crops.
- Preserve the recorded gamma mapping and existing assets of older sessions.
- Document input-processing provenance and the expert-review development workflow.
  Detection accuracy and final acceptance policies require independent validation.

## Initial development toolkit

- Native uint16 reading, descriptive measurements, axial candidate screening,
  source-bound crops and provider-independent investigation interfaces.
