import json
import numpy as np
import pandas as pd

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

def convert_logs_text_to_normalized_df(logs_text: str, debug: bool = False) -> pd.DataFrame:
    """
    Recebe o texto bruto de logs (.log com JSON lines, possivelmente no formato EMA)
    e retorna um DataFrame normalizado no formato legado, com colunas:

        timestamp, user, page, message, __line__, type, selected_agent, model

    Lógica:
      - parseia JSON lines → df_raw;
      - se df_raw já tiver todas as colunas esperadas → usa diretamente (formato legado);
      - caso contrário, tenta inferir colunas de user/model/agent etc. via heurística (formato novo/EMA).

    Além disso, adiciona em df.attrs:
        df.attrs["log_format"] = "legacy" ou "ema"
        df.attrs["conversion_applied"] = bool
    """

    # 1) Parse genérico: JSON por linha
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
        if debug:
            st.write("⚠️ Nenhuma linha JSON válida encontrada no log.")
        cols = ["timestamp", "user", "page", "message", "__line__", "type", "selected_agent", "model"]
        df_empty = pd.DataFrame(columns=cols)
        df_empty.attrs["log_format"] = "unknown"
        df_empty.attrs["conversion_applied"] = False
        return df_empty

    df_raw = pd.DataFrame(rows)
    expected = ["timestamp", "user", "page", "message", "__line__", "type", "selected_agent", "model"]

    # 2) Detecta formato
    is_legacy = set(expected).issubset(df_raw.columns)

    if debug:
        st.markdown("### 🔎 Detecção de formato de log")
        st.write("Colunas brutas:", list(df_raw.columns))
        if is_legacy:
            st.success("Formato detectado: **legado (antigo EVA)**. Nenhuma conversão de formato será aplicada.")
        else:
            st.warning("Formato detectado: **novo/EMA**. Será aplicada conversão para o formato legado.")

    # 3) Caso trivial: já está no formato legado
    if is_legacy:
        df = df_raw.copy()
    else:
        # 4) Formato "novo"/EMA → heurística de colunas
        if debug:
            st.write("♻️ Convertendo logs EMA para formato legado (heurística de colunas)...")

        df = df_raw.copy()

        ts_col    = _find_column_by_substring(df, ["timestamp", "time", "created_at"])
        user_col  = _find_column_by_substring(df, ["user", "email", "actor"])
        page_col  = _find_column_by_substring(df, ["page", "screen", "context"])
        msg_col   = _find_column_by_substring(df, ["message", "content", "text"])
        type_col  = _find_column_by_substring(df, ["type", "event_type", "kind"])
        agent_col = _find_column_by_substring(df, ["selected_agent", "agent", "assistant", "bot"])
        model_col = _find_column_by_substring(df, ["model", "llm_model"])

        if debug:
            st.markdown("#### Mapeamento de colunas inferido")
            st.write(f"  timestamp      ← {ts_col}")
            st.write(f"  user           ← {user_col}")
            st.write(f"  page           ← {page_col}")
            st.write(f"  message        ← {msg_col}")
            st.write(f"  type           ← {type_col}")
            st.write(f"  selected_agent ← {agent_col}")
            st.write(f"  model          ← {model_col}")

        df_norm = pd.DataFrame()
        df_norm["timestamp"]      = df[ts_col]    if ts_col    else np.nan
        df_norm["user"]           = df[user_col]  if user_col  else np.nan
        df_norm["page"]           = df[page_col]  if page_col  else "chat"
        df_norm["message"]        = df[msg_col]   if msg_col   else ""
        df_norm["__line__"]       = df["__line__"] if "__line__" in df.columns else pd.Series(range(1, len(df) + 1))
        df_norm["type"]           = df[type_col]  if type_col  else np.nan
        df_norm["selected_agent"] = df[agent_col] if agent_col else np.nan
        df_norm["model"]          = df[model_col] if model_col else np.nan

        df = df_norm

    # 5) Garante todas as colunas esperadas
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # 6) Normalizações de tipo / formatação
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["user"] = df["user"].astype(str).str.strip().str.lower()
    df["page"] = df["page"].astype(str).str.strip().str.lower()
    df["selected_agent"] = df["selected_agent"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    # 7) Anotar metadados sobre o formato
    df.attrs["log_format"] = "legacy" if is_legacy else "ema"
    df.attrs["conversion_applied"] = (not is_legacy)

    if debug:
        st.markdown("#### ✅ DataFrame normalizado (formato legado)")
        st.dataframe(df.head())

    return df

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
