"""The five team averages and exactly 13 predictors in Appendix B."""

import pandas as pd

from .preprocessing import (
    PLAYER_METRICS, finite_numeric, require_columns, validate_names,
)

TEAM_FEATURES = ("avg_goals", "avg_assists", "avg_xg", "avg_age", "avg_minutes")
DIFF_FEATURES = TEAM_FEATURES[:3]
FEATURE_COLUMNS = (
    [f"home_{name}" for name in TEAM_FEATURES]
    + [f"away_{name}" for name in TEAM_FEATURES]
    + [f"diff_{name}" for name in DIFF_FEATURES]
)


def build_team_strength(players: pd.DataFrame) -> pd.DataFrame:
    require_columns(players, ("team", *PLAYER_METRICS))
    validate_names(players, ("team",))
    players = finite_numeric(players, PLAYER_METRICS, allow_missing=True)
    strength = players.groupby("team", as_index=False)[list(PLAYER_METRICS)].mean()
    strength = strength.rename(columns=dict(zip(PLAYER_METRICS, TEAM_FEATURES)))
    # A missing player value is skipped for that column only, never imputed.
    # A wholly missing team statistic cannot be used by this pipeline.
    return finite_numeric(strength, TEAM_FEATURES)


def build_match_features(matches: pd.DataFrame, team_strength: pd.DataFrame) -> pd.DataFrame:
    """Preserve match row order and share the same feature builder at inference."""
    require_columns(matches, ("home_team", "away_team"))
    validate_names(matches, ("home_team", "away_team"))
    require_columns(team_strength, ("team", *TEAM_FEATURES))
    validate_names(team_strength, ("team",))
    if team_strength["team"].duplicated().any():
        raise ValueError("team_strength must have exactly one row per team.")
    strength = finite_numeric(team_strength, TEAM_FEATURES).set_index("team")
    unknown = sorted(
        (set(matches["home_team"]) | set(matches["away_team"])) - set(strength.index)
    )
    if unknown:
        raise ValueError(f"Teams absent from player statistics after team_map: {unknown}")
    features = pd.DataFrame(index=matches.index)
    for side in ("home", "away"):
        for name in TEAM_FEATURES:
            features[f"{side}_{name}"] = matches[f"{side}_team"].map(strength[name])
    for name in DIFF_FEATURES:
        features[f"diff_{name}"] = features[f"home_{name}"] - features[f"away_{name}"]
    return features[FEATURE_COLUMNS]
