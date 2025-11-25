import math
import numpy as np
import pandas as pd


def _log_normalize(s: pd.Series) -> pd.Series:
    """Log-normalize a positive series to [0, 1]."""
    s = s.astype(float)
    if s.empty:
        return pd.Series([], dtype=float)
    maxv = float(s.max())
    if maxv <= 0 or math.isnan(maxv):
        return pd.Series(0.0, index=s.index, dtype=float)
    return np.log1p(s) / math.log1p(maxv)


def compute_scores_from_df(df: pd.DataFrame, alpha: float = 0.15) -> pd.DataFrame:
    """
    Given a filtered logs DataFrame with columns:
        ['timestamp', 'user', 'page', '__line__', 'type', 'selected_agent', 'model', 'w']
    compute messages, unique_users, HHI, diversity, scoreH and final score
    for each selected_agent.

    alpha controls how much diversity influences the final score.
    """
    df = df.copy()

    # If weights are missing, default to 1.0 per row
    if "w" not in df.columns:
        df["w"] = 1.0

    # --- clean / validate agent + model columns ---
    sel = df["selected_agent"].astype(str).str.strip()
    mod = df["model"].astype(str).str.strip()

    # valid if not empty and not the literal "nan"
    mask_valid_agent = sel.ne("") & sel.str.lower().ne("nan")
    mask_valid_model = mod.ne("") & mod.str.lower().ne("nan")

    df = df[mask_valid_agent & mask_valid_model].copy()
    if df.empty:
        # nothing to score
        return pd.DataFrame(
            columns=["selected_agent", "messages", "unique_users",
                     "hhi", "diversity", "scoreH", "score"]
        )

    # ensure selected_agent column is the cleaned one
    df["selected_agent"] = sel.loc[df.index]

    # --- aggregate by agent ---
    grouped = df.groupby("selected_agent")

    # total weighted messages
    messages = grouped["w"].sum()

    # unique users per agent
    unique_users = grouped["user"].nunique()

    # HHI / diversity
    per_user_w = df.groupby(["selected_agent", "user"])["w"].sum()
    totals = per_user_w.groupby(level=0).sum()
    shares = per_user_w / totals.reindex(per_user_w.index.get_level_values(0)).values
    hhi = shares.pow(2).groupby(level=0).sum().reindex(messages.index, fill_value=1.0)
    diversity = (1.0 - hhi).clip(0.0, 1.0)

    # --- combine into scores ---
    norm_msgs = _log_normalize(messages)
    norm_users = _log_normalize(unique_users.astype(float))

    denom = norm_msgs + norm_users
    scoreH = pd.Series(0.0, index=messages.index, dtype=float)
    mask = denom > 0
    scoreH[mask] = 2.0 * norm_msgs[mask] * norm_users[mask] / denom[mask]

    score = scoreH * (alpha + (1.0 - alpha) * diversity)

    df_scores = pd.DataFrame(
        {
            "selected_agent": messages.index,
            "messages": messages.values,
            "unique_users": unique_users.values,
            "hhi": hhi.values,
            "diversity": diversity.values,
            "scoreH": scoreH.values,
            "score": score.values,
        }
    ).reset_index(drop=True)

    # sort descending by score so the most used/diverse agents are at the top
    df_scores = df_scores.sort_values("score", ascending=False).reset_index(drop=True)
    return df_scores
