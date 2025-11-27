import json
import numpy as np
import pandas as pd
from typing import Tuple

def parse_log_text_to_dataframe(logs_text: str) -> pd.DataFrame:
    """
    Converte um texto de .log em um DataFrame genérico,
    considerando cada linha como um JSON.
    """
    rows = []
    for i, line in enumerate(logs_text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            # Linha não é JSON válido → ignora
            continue
        if not isinstance(obj, dict):
            obj = {"value": obj}
        obj.setdefault("__line__", i)
        rows.append(obj)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def _find_column_by_substring(df: pd.DataFrame, substrings) -> str | None:
    """
    Procura a primeira coluna cujo nome contenha algum dos substrings indicados.
    Retorna o nome da coluna ou None se nada for encontrado.
    """
    substrings = [s.lower() for s in substrings]
    for col in df.columns:
        name = str(col).lower()
        if any(sub in name for sub in substrings):
            return col
    return None

def convert_logs_text_to_normalized_df(logs_text: str, debug: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Detect whether logs are in the 'old/legacy EVA' format (flat log lines with
    timestamp, user, page, message, etc.) or 'new EMA' format (with multi-agent
    metadata), and always return a normalized DataFrame with the legacy schema:

        ['timestamp', 'user', 'page', '__line__', 'type', 'selected_agent', 'model']

    This function is UI-agnostic: it does NOT call streamlit.
    It only returns the DataFrame and a human-readable format label.
    """
    lines = [ln for ln in logs_text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=[
            "timestamp", "user", "page", "__line__", "type", "selected_agent", "model"
        ]), "vazio"

    # helper: parse a JSON line safely
    def _safe_json(line):
        try:
            return json.loads(line)
        except Exception:
            return None

    first_obj = None
    for ln in lines:
        obj = _safe_json(ln)
        if isinstance(obj, dict):
            first_obj = obj
            break

    if first_obj is None:
        # Could not parse any line as JSON
        df_empty = pd.DataFrame(columns=[
            "timestamp", "user", "page", "__line__", "type", "selected_agent", "model"
        ])
        return df_empty, "desconhecido"

    # Heuristic to detect new EMA format (multi-agent)
    is_new = False
    if "agents" in first_obj or "agent_name" in first_obj or "agent_id" in first_obj:
        is_new = True
    if "log_version" in first_obj and str(first_obj["log_version"]).startswith("ema"):
        is_new = True

    # ------------------------------------------------------------------
    # Legacy format → direct normalization
    # ------------------------------------------------------------------
    if not is_new:
        records = []
        for idx, ln in enumerate(lines, start=1):
            obj = _safe_json(ln)
            if not isinstance(obj, dict):
                continue

            timestamp = obj.get("timestamp")
            user = obj.get("user") or obj.get("username")
            page = obj.get("page")
            _type = obj.get("type")
            selected_agent = obj.get("selected_agent") or obj.get("agent_name")
            model = obj.get("model")

            records.append({
                "timestamp": timestamp,
                "user": user,
                "page": page,
                "__line__": idx,
                "type": _type,
                "selected_agent": selected_agent,
                "model": model,
            })

        df = pd.DataFrame.from_records(records)
        # normalize timestamp, but keep tz info if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df, "legado (antigo EVA)"

    # ------------------------------------------------------------------
    # New EMA format → convert to legacy schema
    # ------------------------------------------------------------------
    records = []

    for idx, ln in enumerate(lines, start=1):
        obj = _safe_json(ln)
        if not isinstance(obj, dict):
            continue

        timestamp = obj.get("timestamp")
        user = obj.get("user") or obj.get("username")
        page = obj.get("page") or obj.get("context")  # adjust if needed
        _type = obj.get("type")

        # EMA often nests agent info; adapt to your log structure:
        # ex.: {"selected_agent": {"name": "...", "id": "...", "model": "..."}}
        selected_agent = None
        model = None

        if isinstance(obj.get("selected_agent"), dict):
            sel = obj["selected_agent"]
            selected_agent = sel.get("name") or sel.get("agent_name")
            model = sel.get("model")
        else:
            selected_agent = obj.get("selected_agent") or obj.get("agent_name")
            model = obj.get("model")

        records.append({
            "timestamp": timestamp,
            "user": user,
            "page": page,
            "__line__": idx,
            "type": _type,
            "selected_agent": selected_agent,
            "model": model,
        })

    df = pd.DataFrame.from_records(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df, "novo EMA (convertido para legado)"

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

    # total weighted messages
    messages = grouped["w"].sum()
import json
from datetime import datetime, date

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
