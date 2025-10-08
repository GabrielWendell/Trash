#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA Agent Recommender
---------------------
End-to-end pipeline to produce overall and per-model Top-N agent recommendations
from EVA access logs converted to CSV.

Features
========
- Handles mixed schemas (7 vs 8 columns). Ensures a canonical set of columns:
  ['timestamp','user','page','message','__line__','type','selected_agent','model']
- Enforces policy filters:
    * drop rows with page == 'landing'
    * drop rows where selected_agent is NaN
    * drop rows where model is NaN
- Robust CSV reading (python engine, dtype=str where suitable) and timestamp parsing.
- Optional recency weighting with exponential decay.
- Scoring: log-normalized + harmonic mean (messages vs unique users) with a diversity
  penalty derived from HHI (Herfindahl–Hirschman Index).
- Emits overall and per-model Top-N JSONs containing key metrics.
- Optional enrichment from agent registry (if available) using s3_base.AgentManager.
- Verbose diagnostics (row drops, schema mix, concentration distribution, etc.).

CLI
===
Example:
    python recommend_agents.py \
        --logs-dir logs_csv \
        --out-dir results \
        --topk 10 \
        --alpha 0.15 \
        --decay 0.0 \
        --per-model \
        --verbose

Notes
=====
- This script expects the CSVs produced by your logs-to-CSV converter.
- If you want per-session "accesses", you can enable sessionization with
  --session-gap-mins (e.g. 30). By default, we report messages (=rows) only.
- If s3_base.py is present and AWS credentials are configured, the script will
  try to enrich each agent with owner/name from S3; otherwise it gracefully
  continues without enrichment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional import: agent registry (graceful fallback if missing)
AGENT_REGISTRY_AVAILABLE = False
try:
    from s3_base import AgentManager  # type: ignore
    AGENT_REGISTRY_AVAILABLE = True
except Exception:
    AGENT_REGISTRY_AVAILABLE = False


REQUIRED_COLUMNS = [
    "timestamp",
    "user",
    "page",
    "message",
    "__line__",
    "type",
    "selected_agent",
    "model",
]

TIMESTAMP_FMT = "%Y-%m-%d-%H-%M-%S"  # e.g., 2025-09-01-08-43-38


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EVA Agent Recommender")
    parser.add_argument("--logs-dir", required=True, type=str,
                        help="Directory containing CSV logs (already converted from .log)")
    parser.add_argument("--out-dir", required=True, type=str,
                        help="Directory to write output JSON files")
    parser.add_argument("--topk", type=int, default=10, help="Top-K agents to emit")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Diversity penalty floor (0=no floor, 1=no penalty)")
    parser.add_argument("--decay", type=float, default=0.0,
                        help="Recency exponential decay per day (0 disables)")
    parser.add_argument("--per-model", action="store_true",
                        help="Also compute per-model leaderboards")
    parser.add_argument("--session-gap-mins", type=int, default=0,
                        help="If >0, compute per-user sessions as accesses using this gap. 0=disabled")
    parser.add_argument("--bucket", type=str, default=None,
                        help="(Optional) S3 bucket for agent registry enrichment")
    parser.add_argument("--user-email", type=str, default=None,
                        help="(Optional) Current user email for AgentManager context")
    parser.add_argument("--verbose", action="store_true",
                        help="Print diagnostics")
    return parser.parse_args()


def _ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV with a tolerant parser and return a DataFrame of strings when possible.
    We avoid dtype inference pitfalls by loading as object and converting later as needed.
    """
    try:
        df = pd.read_csv(
            path,
            engine="python",
            dtype=str,
            keep_default_na=True,
            na_values=["", "None", "null", "NaN", "nan"],
            on_bad_lines="skip",
        )
    except TypeError:
        # pandas<2.0 compatibility (on_bad_lines keyword difference)
        df = pd.read_csv(
            path,
            engine="python",
            dtype=str,
            keep_default_na=True,
            na_values=["", "None", "null", "NaN", "nan"],
            error_bad_lines=False,  # deprecated in new pandas
            warn_bad_lines=True,
        )
    # Normalize columns: if model is missing, add it
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    # Keep only required columns, in order
    df = df[REQUIRED_COLUMNS]
    return df


def _parse_timestamp_col(df: pd.DataFrame) -> pd.Series:
    def _parse_one(x: str) -> pd.Timestamp:
        if pd.isna(x):
            return pd.NaT
        x = str(x).strip()
        try:
            return pd.to_datetime(x, format=TIMESTAMP_FMT, utc=True)
        except Exception:
            # Fallback: attempt auto-parse
            try:
                return pd.to_datetime(x, utc=True)
            except Exception:
                return pd.NaT
    return df["timestamp"].apply(_parse_one)


def _policy_filter(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply bank policy filters and report drops.
    Drops rows where:
      - page == 'landing'
      - selected_agent is NaN
      - model is NaN
    """
    stats = {}
    n0 = len(df)

    # page != landing
    mask = df["page"].astype(str).str.strip().str.lower() == "landing"
    stats["dropped_landing"] = int(mask.sum())
    df = df.loc[~mask]

    # drop NaN selected_agent
    mask = df["selected_agent"].isna()
    stats["dropped_selected_agent_nan"] = int(mask.sum())
    df = df.loc[~mask]

    # drop NaN model
    mask = df["model"].isna()
    stats["dropped_model_nan"] = int(mask.sum())
    df = df.loc[~mask]

    stats["kept_after_filters"] = int(len(df))
    stats["dropped_total"] = int(n0 - len(df))

    if verbose:
        print("[FILTER] Drops:", json.dumps(stats, indent=2))
    return df, stats


def _standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    # canonicalize user emails and agent/model identifiers
    df["user"] = df["user"].astype(str).str.strip().str.lower()
    df["selected_agent"] = df["selected_agent"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    return df


def _add_recency_weight(df: pd.DataFrame, decay: float) -> pd.DataFrame:
    if decay <= 0.0:
        df["w"] = 1.0
        return df
    now = pd.Timestamp.utcnow()
    age_days = (now - df["timestamp"]).dt.total_seconds() / 86400.0
    df["w"] = np.exp(-decay * age_days.astype(float))
    return df


def _sessionize_accesses(
    df: pd.DataFrame, gap_minutes: int
) -> Optional[pd.DataFrame]:
    """Compute per-(user, agent) sessions using a time gap heuristic.
    Returns a DataFrame with columns: selected_agent, user, sessions
    """
    if gap_minutes <= 0:
        return None

    df2 = df.sort_values(["selected_agent", "user", "timestamp"]).copy()
    grp = df2.groupby(["selected_agent", "user"], sort=False)

    def count_sessions(g: pd.DataFrame) -> int:
        ts = g["timestamp"].values
        if len(ts) == 0:
            return 0
        # Count a new session whenever gap > threshold
        gaps = (g["timestamp"].diff().dt.total_seconds() / 60.0).fillna(float("inf"))
        return int((gaps > gap_minutes).sum())

    sessions = grp.apply(count_sessions).rename("sessions").reset_index()
    return sessions


# --------------------------------------------------------------------------------------
# Aggregation & Scoring
# --------------------------------------------------------------------------------------

def _compute_hhi_from_weighted_counts(wcounts: pd.Series) -> float:
    total = float(wcounts.sum())
    if total <= 0.0:
        return 1.0  # degenerate; treat as fully concentrated
    shares = (wcounts / total).astype(float)
    return float((shares ** 2).sum())


def _aggregate_agent_metrics(
    df: pd.DataFrame,
    session_gap_mins: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return DataFrame indexed by selected_agent with columns:
    messages, unique_users, hhi, diversity, accesses(optional)
    """
    # messages: weighted count of rows
    messages = df.groupby("selected_agent")["w"].sum().rename("messages")

    # unique users
    unique_users = df.groupby("selected_agent")["user"].nunique().rename("unique_users")

    # HHI: compute per-agent per-user weighted counts
    per_user_w = (
        df.groupby(["selected_agent", "user"])  # type: ignore
        ["w"].sum()
        .rename("wcount")
    )

    hhi = (
        per_user_w
        .groupby(level=0)
        .apply(_compute_hhi_from_weighted_counts)
        .rename("hhi")
        .astype(float)
    )
    diversity = (1.0 - hhi).clip(0.0, 1.0).rename("diversity")

    out = pd.concat([messages, unique_users, hhi, diversity], axis=1)

    # Optional: accesses via sessionization
    if session_gap_mins > 0:
        sessions_df = _sessionize_accesses(df, session_gap_mins)
        if sessions_df is not None and len(sessions_df):
            accesses = (
                sessions_df.groupby("selected_agent")["sessions"].sum().rename("accesses")
            )
            out = out.join(accesses, how="left")
        else:
            out["accesses"] = np.nan
    return out.fillna({"messages": 0.0, "unique_users": 0, "hhi": 1.0, "diversity": 0.0})


def _log_normalize(series: pd.Series) -> pd.Series:
    maxv = float(series.max()) if len(series) else 0.0
    if maxv <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return np.log1p(series.astype(float)) / math.log1p(maxv)


def _harmonic_mean(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = (a + b)
    hm = 2.0 * a * b / denom
    hm[denom == 0.0] = 0.0
    return hm


def _score_agents(df_metrics: pd.DataFrame, alpha: float) -> pd.DataFrame:
    # Log-normalize messages and unique_users
    na = _log_normalize(df_metrics["messages"])  # in [0,1]
    nu = _log_normalize(df_metrics["unique_users"])  # in [0,1]

    scoreH = _harmonic_mean(na, nu)
    diversity = df_metrics["diversity"].astype(float).clip(0.0, 1.0)
    # Diversity-aware penalty with floor alpha
    score = scoreH * (alpha + (1.0 - alpha) * diversity)

    out = df_metrics.copy()
    out["n_messages_log_norm"] = na.values
    out["n_unique_users_log_norm"] = nu.values
    out["scoreH"] = scoreH.values
    out["score"] = score.values
    return out


# --------------------------------------------------------------------------------------
# Enrichment (optional)
# --------------------------------------------------------------------------------------

def _maybe_enrich_with_registry(
    df_ranked: pd.DataFrame,
    bucket: Optional[str],
    user_email: Optional[str],
    verbose: bool = False,
) -> pd.DataFrame:
    """Attempt to enrich each agent with name/owner via s3_base.AgentManager.
    Falls back gracefully if not available or if lookups fail.
    """
    if not (AGENT_REGISTRY_AVAILABLE and bucket and user_email):
        if verbose:
            print("[ENRICH] Registry not available or bucket/user not provided. Skipping.")
        df_ranked["agent_name"] = df_ranked.index.astype(str)
        df_ranked["owner_email"] = None
        return df_ranked

    try:
        am = AgentManager(bucket=bucket, user_mail=user_email)
    except Exception as e:
        if verbose:
            print(f"[ENRICH] Failed to init AgentManager: {e}. Skipping enrichment.")
        df_ranked["agent_name"] = df_ranked.index.astype(str)
        df_ranked["owner_email"] = None
        return df_ranked

    names = []
    owners = []
    for agent_id in df_ranked.index.astype(str):
        name = agent_id
        owner = None
        try:
            ag = am.get_agent(agent_id)
            if hasattr(ag, "nome_agente") and isinstance(ag.nome_agente, str):
                name = ag.nome_agente
            try:
                owner = am.get_agent_owner(agent_id)
            except Exception:
                owner = None
        except Exception:
            pass
        names.append(name)
        owners.append(owner)

    df_ranked["agent_name"] = names
    df_ranked["owner_email"] = owners
    return df_ranked


# --------------------------------------------------------------------------------------
# Main compute
# --------------------------------------------------------------------------------------

def _load_all_csvs(logs_dir: Path, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, int]]:
    paths = sorted(Path(logs_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {logs_dir}")

    dfs = []
    counts = {"files": 0, "with_model_col": 0, "without_model_col": 0}

    for p in paths:
        df = _read_csv_robust(p)
        counts["files"] += 1
        if df["model"].isna().all():
            counts["without_model_col"] += 1
        else:
            counts["with_model_col"] += 1
        dfs.append(df)

    big = pd.concat(dfs, ignore_index=True)
    if verbose:
        print("[LOAD] CSV stats:")
        print(json.dumps(counts, indent=2))
        print(f"[LOAD] Total rows raw: {len(big)}")
    return big, counts


def _prepare_dataframe(df: pd.DataFrame, decay: float, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    df["timestamp"] = _parse_timestamp_col(df)
    df = _standardize_strings(df)

    # Apply filters
    df, drop_stats = _policy_filter(df, verbose=verbose)

    # Remove rows with invalid timestamp after filtering (rare)
    invalid_ts = df["timestamp"].isna()
    if invalid_ts.any():
        drop_stats["dropped_invalid_timestamp"] = int(invalid_ts.sum())
        df = df.loc[~invalid_ts]

    # Recency weight
    df = _add_recency_weight(df, decay)

    if verbose:
        print(f"[CLEAN] Rows after all filters: {len(df)}")
    return df, drop_stats


def _compute_leaderboard(
    df: pd.DataFrame,
    alpha: float,
    topk: int,
    session_gap_mins: int = 0,
    bucket: Optional[str] = None,
    user_email: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    metrics = _aggregate_agent_metrics(df, session_gap_mins=session_gap_mins, verbose=verbose)
    scored = _score_agents(metrics, alpha=alpha)
    ranked = scored.sort_values("score", ascending=False)
    ranked = _maybe_enrich_with_registry(ranked, bucket=bucket, user_email=user_email, verbose=verbose)
    return ranked.head(topk)


def _emit_json(df_ranked: pd.DataFrame, out_path: Path, include_model: Optional[str] = None) -> None:
    cols = [
        "agent_name",  # enriched or fallback to id
        "owner_email",
        "messages",
        "unique_users",
        "hhi",
        "diversity",
        "score",
    ]

    # Ensure index (agent_id) is preserved
    out_df = df_ranked.copy()
    out_df.insert(0, "agent_id", out_df.index.astype(str))
    if include_model is not None:
        out_df.insert(1, "model", include_model)

    # Reorder columns if present
    final_cols = [c for c in ["agent_id", "agent_name", "owner_email", "model", "messages", "unique_users", "hhi", "diversity", "score"] if c in out_df.columns]

    records = [
        {
            k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v)
            for k, v in row.items()
        }
        for row in out_df[final_cols].to_dict(orient="records")
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    # Load
    df_raw, load_stats = _load_all_csvs(logs_dir, verbose=args.verbose)

    # Prepare
    df, drop_stats = _prepare_dataframe(df_raw, decay=args.decay, verbose=args.verbose)

    if args.verbose:
        print("[META] After filtering:")
        print(f"       unique agents: {df['selected_agent'].nunique()}")
        print(f"       unique users : {df['user'].nunique()}")
        print(f"       unique models: {df['model'].nunique()}")

    # Overall leaderboard (all models mixed)
    top_overall = _compute_leaderboard(
        df,
        alpha=args.alpha,
        topk=args.topk,
        session_gap_mins=args.session_gap_mins,
        bucket=args.bucket,
        user_email=args.user_email,
        verbose=args.verbose,
    )
    _emit_json(top_overall, out_dir / "recommendations_top10_overall.json")

    # Per-model leaderboards
    if args.per_model:
        for model, dfm in df.groupby("model"):
            top_m = _compute_leaderboard(
                dfm,
                alpha=args.alpha,
                topk=args.topk,
                session_gap_mins=args.session_gap_mins,
                bucket=args.bucket,
                user_email=args.user_email,
                verbose=args.verbose,
            )
            safe_model = str(model).replace(os.sep, "_").replace(" ", "_")
            _emit_json(top_m, out_dir / f"recommendations_top10_model={safe_model}.json", include_model=str(model))

    # Diagnostics dump
    if args.verbose:
        diag = {
            "load": load_stats,
            "drops": drop_stats,
            "final_rows": int(len(df)),
        }
        with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        print("[DONE] Wrote:")
        print("  - recommendations_top10_overall.json")
        if args.per_model:
            print("  - recommendations_top10_model=*.json")
        print("  - diagnostics.json")


if __name__ == "__main__":
    main()
