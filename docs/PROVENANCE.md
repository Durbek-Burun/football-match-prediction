# Source traceability

Sources inspected:

1. `Эркинов_Дурбек_ВКР.pdf`, 20 pages, Tashkent, 2026, by Эркинов Дурбек Кахрамон угли.
2. User-supplied `football_match_prediction_project.rar`: five source modules,
   five CSVs, README, project report, and requirements. Model and notebook
   directories were empty before validation.
3. Separately supplied match and team-strength CSVs, byte-identical to the archive.

| Implemented element | Evidence |
|---|---|
| Player statistics and match results, season 2024/2025 | Thesis §2.1, p. 10; Appendix A, p. 19; supplied CSVs |
| `Squad.replace(team_map)` and five unweighted player means | Appendix A, p. 19; rebuilt means match the archive |
| Actual 96-entry club mapping | Archive `team_map_2024_2025.csv` |
| Column-wise omission of missing player ages | Appendix A pandas mean; matches saved strengths for all teams |
| Exactly 13 predictors and `target_result` | Appendix B, p. 19 |
| Home-minus-away goals, assists and xG | Appendix C, p. 20 |
| Stratified 80/20 split, `random_state=42` | Appendix B; archive `train_model.py` |
| 700 iterations, 0.04 learning rate, depth 6, MultiClass, Accuracy | §3.3, p. 14; Appendix B; archive training code |
| Accuracy, confusion matrix, class report, importance | §4.1, p. 15; archive training code |
| Saved model, class and probability prediction | §3.3–3.4, p. 14; Appendix C; archive prediction code |
| Exclusion of incomplete match predictors/target | Archive `train_model.py`, `dropna(subset=needed_columns)` |
| Approximate 60% historical claim | Thesis §4.2, p. 15, and conclusion, p. 17 |

## Thesis and archive differ

The thesis appendix specifies 13 features and no explicit CatBoost random seed.
The archive implements 24 features: eight home, eight away, eight differences;
it adds xAG, matches played and starts, plus differences for all eight team
statistics. It also sets `random_seed=42`. Main `src/train.py` retains the thesis
method. The unmodified archive script was run separately; its measured report is
`reports/archive24_metrics.json`. These are distinct experimental configurations.

The original README's illustrative probabilities are not treated as measurements.
All charts in this repository come from the newly executed 13-feature run.

## Repository engineering additions

The command-line interface, archive adapter, strict schema checks, feature-order
validation, shared train/inference builder, output protection, input hashes,
version records, explicit class order, report exports and charts were added for
reproducibility. They are not claimed to be verbatim thesis code. Suppressing
CatBoost auxiliary output with `allow_writing_files=False` changes file output,
not the predictive feature set. No early stopping or test-set selection is used.

Exact historical package versions, source acquisition scripts, fuzzy-match
thresholds and raw upstream dataset provenance remain unavailable. The attached
player CSV already contains joined tables; this repository verifies and consumes
its five relevant player metrics rather than inventing how the upstream joins
were performed. Temporal-leakage concerns are a review observation, not a thesis
measurement or an automatically corrected part of this implementation.
