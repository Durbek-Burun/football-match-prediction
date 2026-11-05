# Validation status

Checked on 2026-11-03.

- All six Python modules passed parsing and bytecode compilation.
- Dependencies were installed in an isolated local environment; the actual
  versions are recorded in `requirements-lock.txt` and `reports/metadata.json`.
- Preparation ran on the supplied CSVs: 2,854 player rows, 96 teams, 1,752 retained
  matches and 189 audited exclusions. Mappings, team means and supplied match
  features passed consistency checks within `1e-10` tolerance.
- Training completed all 700 iterations for the 13-feature model: accuracy
  **51.28%** on 351 held-out matches. The original archive's unmodified 24-feature
  training code completed separately: **49.86%**.
- Train and test match IDs were verified identical between the two runs.
- Reloaded model classes and probabilities reproduced all saved test outputs.
  Every probability row sums to one within `1e-12` tolerance.
- The documented Arsenal FC / Chelsea FC prediction command ran successfully.
  Its measured output is in `reports/prediction_example.json`; it is an interface
  demonstration using the seasonal model, not a forecast for a dated future game.
- Unknown teams, duplicate team-strength rows, infinite features, and a wholly
  missing team statistic raised validation errors. Eight missing player ages
  were preserved until column-wise mean aggregation.
- Reloading the original 24-feature model reproduced its recorded accuracy.
- Internal Markdown links, JSON files and ZIP integrity/content were checked.
  The GitHub-ready ZIP excludes local data, model bundles, caches and environments.

Machine-readable checks and source-code hashes are in
[reports/validation.json](reports/validation.json). Original data-file hashes
are in [reports/preparation_report.json](reports/preparation_report.json).
The two separately supplied CSVs were byte-identical to the archived versions.

The approximately 60% accuracy stated in the thesis was not reproduced. Historical
library versions and the upstream data acquisition/preparation pipeline remain
unknown. Validation covers this recorded local environment and supplied data;
future-match accuracy, probability calibration and other platforms were not tested.
