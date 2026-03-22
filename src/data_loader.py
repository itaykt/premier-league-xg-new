"""
Load StatsBomb open data and build shot-level tables with freeze frames and scoreline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from statsbombpy import sb
except ImportError as e:  # pragma: no cover
    raise ImportError("Install statsbombpy: pip install statsbombpy") from e

# Premier League (men's) in StatsBomb open data — 2015/16 primary season
DEFAULT_COMPETITION_ID = 2
DEFAULT_SEASON_ID = 27


def _shot_mask(df: pd.DataFrame) -> pd.Series:
    if "type_name" in df.columns:
        return df["type_name"] == "Shot"
    if "type" not in df.columns:
        raise ValueError("Events frame must have 'type' or 'type_name'")
    t = df["type"]
    if t.dtype == object:
        return t.apply(
            lambda x: (isinstance(x, dict) and x.get("name") == "Shot") or x == "Shot"
        )
    return t == "Shot"


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def load_matches(
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
) -> pd.DataFrame:
    return sb.matches(competition_id=competition_id, season_id=season_id)


def load_match_events(match_id: int) -> pd.DataFrame:
    return sb.events(match_id=match_id)


def _is_goal_event(row: pd.Series) -> bool:
    typ = row.get("type")
    if typ == "Goal" or (isinstance(typ, dict) and typ.get("name") == "Goal"):
        return True
    if row.get("shot_outcome") == "Goal":
        return True
    if isinstance(typ, dict) and typ.get("name") == "Shot":
        shot = row.get("shot")
        if isinstance(shot, dict):
            out = shot.get("outcome") or {}
            if isinstance(out, dict) and out.get("name") == "Goal":
                return True
    return False


def annotate_goal_diff_at_event(events: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, goal difference for that row's team **before** this event resolves
    (so a goal-scoring shot sees the pre-goal score).
    """
    df = events.sort_values(["period", "index"]).copy()
    if "team_id" not in df.columns and "team" in df.columns:
        df["team_id"] = df["team"].apply(lambda t: t["id"] if isinstance(t, dict) else pd.NA)

    teams = sorted({int(x) for x in df["team_id"].dropna().unique().tolist()})
    if len(teams) < 2:
        df["goal_diff"] = 0
        return df

    t_a, t_b = teams[0], teams[1]
    goals: dict[int, int] = {t_a: 0, t_b: 0}

    diffs: list[int] = []
    for _, row in df.iterrows():
        tid = row.get("team_id")
        if pd.isna(tid):
            diffs.append(0)
            if _is_goal_event(row):
                # Rare: goal without team_id
                pass
            continue
        tid = int(tid)
        opp = t_b if tid == t_a else t_a
        diffs.append(goals[tid] - goals[opp])
        if _is_goal_event(row):
            goals[tid] = goals.get(tid, 0) + 1

    df["goal_diff"] = diffs
    return df


def goal_diff_bucket(gd: float) -> str:
    if gd <= -2:
        return "trail_2plus"
    if gd == -1:
        return "trail_1"
    if gd == 0:
        return "draw"
    if gd == 1:
        return "lead_1"
    return "lead_2plus"


def extract_shots_from_events(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Single match: shot rows with match_id and freeze frame preserved."""
    df = annotate_goal_diff_at_event(events)
    shots = df[_shot_mask(df)].copy()
    shots["match_id"] = match_id

    if "team_id" not in shots.columns and "team" in shots.columns:
        shots["team_id"] = shots["team"].apply(lambda t: t["id"] if isinstance(t, dict) else pd.NA)

    # Normalize freeze frame column name across statsbombpy versions
    ff = _col(shots, "shot_freeze_frame", "freeze_frame")
    if ff is not None:
        shots["freeze_frame"] = ff
    elif "shot" in shots.columns:
        shots["freeze_frame"] = shots["shot"].apply(
            lambda s: s.get("freeze_frame") if isinstance(s, dict) else None
        )

    if "under_pressure" not in shots.columns:
        shots["under_pressure"] = False

    shots["goal_diff_bucket"] = shots["goal_diff"].apply(goal_diff_bucket)

    # Minute normalized 0–1 (90 + stoppage cap)
    if "minute" in shots.columns:
        shots["minute_norm"] = (shots["minute"].clip(0, 100) / 100.0).astype(float)
    else:
        shots["minute_norm"] = 0.5

    return shots


def load_season_shots(
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
    match_ids: list[int] | None = None,
    max_matches: int | None = None,
) -> pd.DataFrame:
    """
    Load all shot events for a competition/season. Optionally limit matches for quick dev runs.
    """
    matches = load_matches(competition_id, season_id)
    mids = matches["match_id"].tolist()
    if match_ids is not None:
        mids = [m for m in mids if m in match_ids]
    if max_matches is not None:
        mids = mids[:max_matches]

    frames: list[pd.DataFrame] = []
    for mid in mids:
        ev = load_match_events(int(mid))
        sh = extract_shots_from_events(ev, int(mid))
        frames.append(sh)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_processed(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def home_away_team_ids_from_events(
    events: pd.DataFrame,
    home_name: str,
    away_name: str,
) -> tuple[int | None, int | None]:
    """
    Map home/away display names from the matches table to team_id values used in events.
    """
    tid_to_name: dict[int, str] = {}
    for _, row in events.iterrows():
        tid = row.get("team_id")
        team = row.get("team")
        if pd.isna(tid):
            continue
        tid_i = int(tid)
        if tid_i not in tid_to_name:
            if isinstance(team, dict):
                tid_to_name[tid_i] = str(team.get("name", "")).strip()
            elif isinstance(team, str):
                tid_to_name[tid_i] = team.strip()

    def norm(s: str) -> str:
        return " ".join(s.split()).strip()

    hn, an = norm(str(home_name)), norm(str(away_name))
    home_id = next((tid for tid, n in tid_to_name.items() if norm(n) == hn), None)
    away_id = next((tid for tid, n in tid_to_name.items() if norm(n) == an), None)
    return home_id, away_id


def build_match_team_id_map(matches: pd.DataFrame) -> dict[int, tuple[int | None, int | None]]:
    """One events fetch per match — use when exporting app data with team ids."""
    out: dict[int, tuple[int | None, int | None]] = {}
    for _, r in matches.iterrows():
        mid = int(r["match_id"])
        ev = load_match_events(mid)
        hi, ai = home_away_team_ids_from_events(ev, str(r["home_team"]), str(r["away_team"]))
        out[mid] = (hi, ai)
    return out


def match_metadata_table(
    competition_id: int = DEFAULT_COMPETITION_ID,
    season_id: int = DEFAULT_SEASON_ID,
) -> pd.DataFrame:
    m = load_matches(competition_id, season_id)
    want = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]
    keep = [c for c in want if c in m.columns]
    return m[keep] if keep else m
