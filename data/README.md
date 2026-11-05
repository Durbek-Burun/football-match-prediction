# Data contract and archive preparation

Inputs were provided in `football_match_prediction_project.rar`. Its `data/`
directory contains five CSVs: players with team statistics, match-level training
data, a team-name mapping, team aggregates, and team match totals. The separate
match and strength CSVs subsequently supplied from Telegram Desktop have the
same SHA-256 hashes as their archive counterparts.

The exact data-provider URLs, original collection process and redistribution
terms remain unverified. This repository does not bundle the original CSVs.
Supply the original archive locally to reproduce the run.

## Prepare the provided archive

```bash
python -m src.prepare_archive --source "PATH/TO/ORIGINAL/data" --output data/prepared
```

The adapter uses the actual 96-row `Squad` → `team` mapping. It verifies the mapped
player teams, reconstructs the five team means, and compares them with the
provided `team_strength_base_2024_2025.csv` within `1e-10` tolerance. It also
compares the reconstructed home/away predictors with the provided match table.

The original training script drops matches missing any required predictor or
target. All 189 Champions League rows lack the supplied home and away statistics;
1,752 domestic-league matches remain. The adapter checks that the original
24-feature and thesis 13-feature completeness masks select exactly the same rows.
It writes `excluded_matches.csv` with IDs and reasons instead of silently hiding
those exclusions. It does not infer missing Champions League features.

`players.csv`, `matches.csv`, `team_map.json`, `excluded_matches.csv` and
`preparation_report.json` are written to a new or empty destination. The public
copy of the preparation report is in `reports/`, with input hashes and counts.
`team_stats_2024_2025.csv` is not needed by the 13-feature model.

## Direct training inputs

Player CSV: `Squad`, `Gls`, `Ast`, `xG`, `Age`, `Min`. Club names must be nonempty
strings. Statistics must be numeric; infinite values are rejected. Missing player
statistics are ignored column by column by the group mean, as in the thesis's
pandas aggregation. No values are imputed, and a wholly missing team statistic
raises an error. In the supplied data, only eight `Age` values are missing among
the five consumed statistics. Other player columns are not predictors.

Match CSV: `home_team`, `away_team`, and `target_result`, whose allowed values are
`Home Win`, `Draw`, `Away Win`. `--target-column` selects a different target name;
`--result-map` accepts an explicit JSON mapping from source labels (strings) to
these names. No target encoding is guessed. Optional `match_id`, `date_utc`, and
`competition_name` are carried into held-out reporting, not used as predictors.

`team_map.json` maps exact player names to match names, preserving names absent
from the mapping (pandas `replace` behavior). The provided example JSON has only
four thesis examples. Use the imported mapping for the real data. No fuzzy
algorithm is invented. Unknown match teams are errors in direct training.

Inputs must already represent the intended season. No automatic season filter,
transfer-row deduplication, minute weighting, or score-based feature engineering
is applied. Source match IDs were unique, and supplied outcome labels agreed
with full-time scores in the validation run.
