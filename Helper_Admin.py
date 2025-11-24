def parse_log_text_to_dataframe(text: str):
    import json
    import pandas as pd

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    return pd.DataFrame(rows)

import numpy as np
import pandas as pd
import math

def _log_normalize(series):
    """Normalize series using log1p scaling."""
    maxv = float(series.max()) if len(series) else 0.0
    if maxv <= 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return np.log1p(series.astype(float)) / math.log1p(maxv)

def compute_scores_from_df(df, alpha=0.15):
    """
    Compute EVA agent scores from filtered logs dataframe.
    Expects columns: selected_agent, user, w (recency weight), etc.
    """
    # Basic aggregation per agent
    grouped = df.groupby("selected_agent")

    messages = grouped["w"].sum()
    unique_users = grouped["user"].nunique()

    # User share distribution → HHI → diversity
    shares_dict = {}
    for ag, subdf in grouped:
        per_user = subdf.groupby("user")["w"].sum()
        tot = per_user.sum()
        if tot <= 0:
            shares_dict[ag] = 1.0
        else:
            shares = (per_user / tot).astype(float)
            shares_dict[ag] = float((shares ** 2).sum())

    hhi = pd.Series(shares_dict)
    diversity = (1.0 - hhi).clip(0, 1)

    # Normalize interactions and users
    na = _log_normalize(messages)
    nu = _log_normalize(unique_users)

    # Harmonic mean
    scoreH = (2.0 * na * nu) / (na + nu)
    scoreH[(na + nu) == 0.0] = 0.0

    # Final score
    score = scoreH * (alpha + (1.0 - alpha) * diversity)

    # Build DataFrame
    return pd.DataFrame({
        "selected_agent": messages.index,
        "messages": messages.values,
        "unique_users": unique_users.values,
        "hhi": hhi.values,
        "diversity": diversity.values,
        "scoreH": scoreH.values,
        "score": score.values,
    })

