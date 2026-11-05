# Measured experiment — 2026-11-03

The main experiment uses the 13-feature thesis implementation and the source
data supplied by the user. No synthetic dataset is used.

| Quantity | Value |
|---|---:|
| Player rows / teams | 2,854 / 96 |
| Source / retained / excluded matches | 1,941 / 1,752 / 189 |
| Training / test matches | 1,401 / 351 |
| Accuracy | 51.28% |
| Macro F1 | 0.4669 |
| Draw precision | 0.3529 |
| Draw recall | 0.2045 |
| Draw F1 | 0.2590 |

Split: `test_size=0.2`, `stratify=y`, `random_state=42`. All 700 iterations are
trained without using the test set for stopping or model selection. The
effective CatBoost seed is recorded in [metadata.json](metadata.json).

Read the full [classification report](classification_report.txt),
[confusion matrix](confusion_matrix.csv), [feature importance](feature_importance.csv),
and [preparation audit](preparation_report.json). Confusion-matrix rows are true
labels and columns are predictions, in `Away Win`, `Draw`, `Home Win` order.

The archive's unmodified 24-feature script achieved **49.86%** on the
same retained matches and test split. Its [separate report](archive24_metrics.json)
records the comparison; it uses model seed 42, while the main run preserves the
thesis's default seed. No performance difference is attributed solely to the
additional predictors.

Full-season averages may include post-match information. The test split is
random, not chronological. The measured accuracy is therefore retrospective;
it does not establish reliability for future games. All 189 Champions League
rows are excluded for missing supplied features, limiting evaluation coverage.
Probabilities have not been calibrated in a separate experiment.

The model correctly classified 18 of the 88 draws, so draw recall is a clear
weakness. Predicting the training-majority class (Home Win) for every test match
would give 147/351 = 41.88% accuracy; the measured model is above this simple
reference. The largest importances in this run are home average age, away average
age, and the xG difference. This numerical ranking is not substituted with the
thesis's qualitative claim about attacking indicators.
