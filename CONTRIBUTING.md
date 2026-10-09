# Contributing to FluoInspect

Keep measurements, artifact interpretation and human decisions separate. A
detector contribution should describe its physical or mathematical basis,
required inputs, units, supported image types and known confounds. Include
meaningful synthetic challenges and evidence from independent reviewed data
before claiming detection performance.

Use synthetic examples in pull requests. Real study data need an independently
established sharing permission and should be hosted separately with appropriate
provenance. Report reproducible method settings and quantitative results instead
of committing local runtime folders.

Run Ruff and pytest for code changes. Reader changes must preserve native pixels,
byte order interpretation and coordinates. Unsupported or missing data must stay
explicit; a failed measurement must not become a zero score or quality acceptance.

Model adapters should implement the backend contract, record model and prompt
versions, use bounded retries and return localized evidence references. Fixed
verification requirements must remain separate from generated instructions.
