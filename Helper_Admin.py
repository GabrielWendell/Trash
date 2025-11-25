import math
import numpy as np
import pandas as pd
import json
import streamlit as st

# ------------------------------------------------------
# Helpers: parsing logs and computing scores
# ------------------------------------------------------

def parse_log_text_to_dataframe(text: str) -> pd.DataFrame:
    """
    Converte o conteúdo de um .log (JSON lines) em DataFrame.
    Supõe um log por linha.
    """
    rows = []
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            obj = {"value": obj}
        obj["__line__"] = i
        rows.append(obj)
    return pd.DataFrame(rows)


def _log_normalize(series: pd.Series) -> pd.Series:
    """Normalização log1p para escalar contagens."""
    maxv = float(series.max()) if len(series) else 0.0
    if maxv <= 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return np.log1p(series.astype(float)) / math.log1p(maxv)


def compute_scores_from_df(df: pd.DataFrame, alpha: float = 0.15) -> pd.DataFrame:
    """
    Calcula score por agente a partir de um DataFrame de logs já filtrado.

    Espera colunas:
      - selected_agent (str)
      - user (str)
      - opcional: w (peso de recência). Se ausente, assume w = 1.0.
    """
    df = df.copy()

    # drop missing or "nan" selected_agent
    sel = df["selected_agent"].astype(str).str.strip()
    mask_valid_agent = sel.ne(""). & sel.str.lower().ne("nan")
    df = df[mask_valid_agent]
    
    # drop missing or "nan" model
    mod = df["model"].astype(str).str.strip()
    mask_valid_model = mod.ne("") & mod.str.lower().ne("nan")
    df = df[mask_valid_model]
    
    # Garante coluna de peso
    if "w" not in df.columns:
        df = df.copy()
        df["w"] = 1.0

    grouped = df.groupby("selected_agent")

    messages = grouped["w"].sum()
    unique_users = grouped["user"].nunique()

    # Diversidade via HHI (Herfindahl-Hirschman Index)
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

    # Normaliza interações e usuários
    na = _log_normalize(messages)
    nu = _log_normalize(unique_users)

    # Média harmônica
    scoreH = (2.0 * na * nu) / (na + nu)
    scoreH[(na + nu) == 0.0] = 0.0

    # Score final
    score = scoreH * (alpha + (1.0 - alpha) * diversity)

    return pd.DataFrame(
        {
            "selected_agent": messages.index,
            "messages": messages.values,
            "unique_users": unique_users.values,
            "hhi": hhi.values,
            "diversity": diversity.values,
            "scoreH": scoreH.values,
            "score": score.values,
        }
    )
