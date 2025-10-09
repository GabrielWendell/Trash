#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA Agents → Enriched JSON (Public/Private)
==========================================

Builds per-agent metadata by cross-matching EVA agent YAMLs with access logs.
Outputs two JSON files (public/private) with enriched fields, plus diagnostics.

Main features
-------------
- Scans a root directory with subfolders: public/, private/ (group/, groups/ ignored).
- Parses per-owner agent YAMLs with robust key fallbacks.
- Loads pre-converted CSV logs (same schema as recommend_agents.py), applies bank filters.
- Matches logs to agents by ID or by agent name (normalized), resolves name collisions.
- Computes per-agent metrics: messages, unique users, HHI/diversity, dominant model.
- Computes scores with the same method as the recommender: log-normalized + harmonic mean ×
  diversity penalty with floor `alpha` (optionally with recency decay on rows).
- Generates succinct description from prompts via a pluggable summarizer (LLM stub + cache),
  with a deterministic fallback when LLM is unavailable.
- Emits per-visibility JSONs and a diagnostics JSON.

CLI
---
Example:
    python enrich_agents.py \
      --agents-root "ChatBots Files/s3_agents_download" \
      --logs-dir logs_csv \
      --out-dir results \
      --alpha 0.15 \
      --decay 0.0 \
      --visibilities public private \
      --verbose

Notes
-----
- Visibility is a *string*: "public" or "private".
- Scores are computed *globally* across all selected visibilities for comparability,
  then outputs are split by visibility. You can change this with --score-scope per-folder.
- This script does not perform network calls. The summarizer is a stub; plug your LLM
  in summarize_prompt_llm() if desired.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
REQUIRED_LOG_COLUMNS = [
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
VALID_VISIBILITIES = {"public", "private"}

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich EVA agents by cross-matching YAMLs with logs")
    p.add_argument("--agents-root", required=True, type=str,
                   help="Root path containing public/, private/ (and possibly group/, groups/)")
    p.add_argument("--logs-dir", required=True, type=str,
                   help="Directory with pre-converted CSV logs (same format used by recommender)")
    p.add_argument("--out-dir", required=True, type=str, help="Directory to save outputs")
    p.add_argument("--visibilities", nargs="*", default=["public", "private"],
                   help="Which visibilities to include (default: public private)")
    p.add_argument("--alpha", type=float, default=0.15,
                   help="Diversity penalty floor for scoring [0..1]")
    p.add_argument("--decay", type=float, default=0.0,
                   help="Recency exponential decay per day (0 disables)")
    p.add_argument("--score-scope", choices=["global", "per-folder"], default="global",
                   help="Compute scoring normalization across all selected folders (global) or per folder")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------

def ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
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
        df = pd.read_csv(
            path,
            engine="python",
            dtype=str,
            keep_default_na=True,
            na_values=["", "None", "null", "NaN", "nan"],
            error_bad_lines=False,  # for pandas<2.0
            warn_bad_lines=True,
        )
    # Canonicalize columns
    for col in REQUIRED_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[REQUIRED_LOG_COLUMNS]
    return df


def parse_timestamp_col(df: pd.DataFrame) -> pd.Series:
    def _parse_one(x: str) -> pd.Timestamp:
        if pd.isna(x):
            return pd.NaT
        x = str(x).strip()
        try:
            return pd.to_datetime(x, format=TIMESTAMP_FMT, utc=True)
        except Exception:
            try:
                return pd.to_datetime(x, utc=True)
            except Exception:
                return pd.NaT
    return df["timestamp"].apply(_parse_one)

# --------------------------------------------------------------------------------------
# Normalization & Filters
# --------------------------------------------------------------------------------------

def empty_like_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip().str.lower()
    return s.isna() | s_str.isin(["", "nan", "none", "null"])


def policy_filter(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats: Dict[str, int] = {}
    n0 = len(df)

    mask = df["page"].astype(str).str.strip().str.lower() == "landing"
    stats["dropped_landing"] = int(mask.sum())
    df = df.loc[~mask]

    mask = empty_like_series(df["selected_agent"])  # invalid agents
    stats["dropped_selected_agent_invalid"] = int(mask.sum())
    df = df.loc[~mask]

    mask = empty_like_series(df["model"])  # invalid models
    stats["dropped_model_invalid"] = int(mask.sum())
    df = df.loc[~mask]

    stats["kept_after_filters"] = int(len(df))
    stats["dropped_total"] = int(n0 - len(df))

    if verbose:
        print("[FILTER]", json.dumps(stats, indent=2))
    return df, stats


def standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df["user"] = df["user"].astype(str).str.strip().str.lower()
    df["selected_agent"] = df["selected_agent"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    return df


def add_recency_weight(df: pd.DataFrame, decay: float) -> pd.DataFrame:
    if decay <= 0.0:
        df["w"] = 1.0
        return df
    now = pd.Timestamp.utcnow()
    age_days = (now - df["timestamp"]).dt.total_seconds() / 86400.0
    df["w"] = np.exp(-decay * age_days.astype(float))
    return df

# --------------------------------------------------------------------------------------
# YAML Registry Builder
# --------------------------------------------------------------------------------------
@dataclass
class AgentRow:
    id: str
    name: str
    owner: str
    prompt: str
    temperature: float
    visibility: str  # "public" or "private"


def safe_read_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        return y
    except Exception:
        return {}


def normalize_email(s: str) -> str:
    return (s or "").strip().lower()


def normalize_name(s: str) -> str:
    # collapse whitespace
    return " ".join((s or "").split()).strip()


def scan_agents(root: Path, visibilities: Iterable[str], verbose: bool = False) -> pd.DataFrame:
    rows: List[AgentRow] = []
    for vis in visibilities:
        if vis not in VALID_VISIBILITIES:
            continue
        vis_dir = root / vis
        if not vis_dir.exists():
            if verbose:
                print(f"[WARN] Visibility dir missing: {vis_dir}")
            continue
        for owner_dir in sorted(vis_dir.glob("*/")):
            owner = normalize_email(owner_dir.name)
            for yml in sorted(owner_dir.glob("*.yaml")):
                y = safe_read_yaml(yml)
                # Fallbacks for keys
                ag_id = (y.get("agent_id") or y.get("id") or y.get("id_agente") or yml.stem or "").strip()
                ag_nm = (y.get("agent_name") or y.get("nome_agente") or y.get("name") or "").strip()
                pr = y.get("prompt") or ""
                tmp = y.get("temp", y.get("temperature", 0.0))
                try:
                    tmp = float(tmp)
                except Exception:
                    tmp = 0.0
                rows.append(AgentRow(
                    id=ag_id,
                    name=normalize_name(ag_nm),
                    owner=owner,
                    prompt=str(pr),
                    temperature=tmp,
                    visibility=vis,
                ))
    df = pd.DataFrame([r.__dict__ for r in rows])
    if verbose:
        print(f"[AGENTS] Loaded {len(df)} agents from {list(visibilities)}")
    return df

# --------------------------------------------------------------------------------------
# Logs Loader
# --------------------------------------------------------------------------------------

def load_and_prepare_logs(logs_dir: Path, decay: float, verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, int]]:
    csvs = sorted(logs_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV logs in {logs_dir}")
    dfs = []
    for p in csvs:
        df = read_csv_robust(p)
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)

    # Parse timestamp then filter to avoid turning NaN into strings
    big["timestamp"] = parse_timestamp_col(big)
    big, drop_stats = policy_filter(big, verbose=verbose)
    big = standardize_strings(big)

    # Remove invalid timestamps
    bad_ts = big["timestamp"].isna()
    if bad_ts.any():
        drop_stats["dropped_invalid_timestamp"] = int(bad_ts.sum())
        big = big.loc[~bad_ts]

    big = add_recency_weight(big, decay)

    if verbose:
        print(f"[LOGS] Rows after filters: {len(big)}")
        print(f"        Unique users : {big['user'].nunique()}")
        print(f"        Unique agents: {big['selected_agent'].nunique()}")
        print(f"        Unique models: {big['model'].nunique()}")
    return big, drop_stats

# --------------------------------------------------------------------------------------
# Matching & Metrics
# --------------------------------------------------------------------------------------

def match_rows_for_agent(df_logs: pd.DataFrame, ag: pd.Series) -> pd.DataFrame:
    """Return subset of df_logs for this agent by ID or by name.
    Priority: match by ID (selected_agent == id), else by name.
    """
    # First try exact ID matches (rare but supported)
    mask_id = df_logs["selected_agent"].str.casefold() == ag["id"].casefold()
    if mask_id.any():
        return df_logs.loc[mask_id]
    # Else try by name (normalized)
    mask_nm = df_logs["selected_agent"].str.casefold() == ag["name"].casefold()
    return df_logs.loc[mask_nm]


def compute_agent_metrics(df_rows: pd.DataFrame) -> Tuple[float, int, float, float, Optional[str]]:
    """Return messages, unique_users, hhi, diversity, dominant_model."""
    if df_rows is None or len(df_rows) == 0:
        return 0.0, 0, 1.0, 0.0, None

    # Weighted messages (each row = one message; use weight if decay enabled)
    messages = float(df_rows["w"].sum())
    unique_users = int(df_rows["user"].nunique())

    per_user_w = (
        df_rows.groupby("user")["w"].sum().rename("wcount")
    )
    total = float(per_user_w.sum())
    if total <= 0.0:
        hhi = 1.0
    else:
        shares = (per_user_w / total).astype(float)
        hhi = float((shares ** 2).sum())
    diversity = max(0.0, min(1.0, 1.0 - hhi))

    # Dominant model: most frequent; tie-break with most recent timestamp
    grp = df_rows.groupby("model")
    counts = grp.size().sort_values(ascending=False)
    if len(counts) == 0:
        dominant_model = None
    else:
        top_count = counts.iloc[0]
        top_models = counts[counts == top_count].index.tolist()
        if len(top_models) == 1:
            dominant_model = str(top_models[0])
        else:
            # tie-break by most recent timestamp per model
            latest_by_model = grp["timestamp"].max()
            dominant_model = str(latest_by_model.loc[top_models].idxmax())

    return messages, unique_users, hhi, diversity, dominant_model

# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

def log_normalize(series: pd.Series) -> pd.Series:
    maxv = float(series.max()) if len(series) else 0.0
    if maxv <= 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return np.log1p(series.astype(float)) / math.log1p(maxv)


def harmonic_mean(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = a + b
    hm = (2.0 * a * b) / denom
    hm[denom == 0.0] = 0.0
    return hm


def score_agents(df_metrics: pd.DataFrame, alpha: float) -> pd.DataFrame:
    na = log_normalize(df_metrics["messages"])  # [0,1]
    nu = log_normalize(df_metrics["unique_users"])  # [0,1]
    scoreH = harmonic_mean(na, nu)
    div = df_metrics["diversity"].astype(float).clip(0, 1)
    score = scoreH * (alpha + (1.0 - alpha) * div)

    out = df_metrics.copy()
    out["n_messages_log_norm"] = na.values
    out["n_unique_users_log_norm"] = nu.values
    out["scoreH"] = scoreH.values
    out["score"] = score.values
    return out

# --------------------------------------------------------------------------------------
# Prompt summarizer (LLM stub + cache)
# --------------------------------------------------------------------------------------

class PromptSummarizer:
    def __init__(self):
        self.cache: Dict[str, str] = {}

    def summarize(self, prompt: str) -> str:
        key = (prompt or "").strip()
        if key in self.cache:
            return self.cache[key]
        if not key:
            desc = "Agent without a defined prompt."
        else:
            # Deterministic fallback: take first sentence-like chunk up to ~160 chars
            desc = self._fallback_summary(key)
        self.cache[key] = desc
        return desc

    @staticmethod
    def _fallback_summary(prompt: str, max_chars: int = 160) -> str:
        text = " ".join(prompt.split())
        # Try to end on a period if possible within max_chars
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        last_dot = cut.rfind(".")
        if last_dot >= 40:  # avoid extremely short fragments
            return cut[: last_dot + 1]
        return cut + "…"

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    agents_root = Path(args.agents_root)
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    visibilities = [v for v in args.visibilities if v in VALID_VISIBILITIES]
    if not visibilities:
        raise ValueError("No valid visibilities specified (use: public, private)")

    # 1) Scan YAML agents
    df_agents = scan_agents(agents_root, visibilities, verbose=args.verbose)
    if df_agents.empty:
        raise RuntimeError("No agents found under the specified visibilities.")

    # 2) Load logs and prepare
    df_logs, drop_stats = load_and_prepare_logs(logs_dir, decay=args.decay, verbose=args.verbose)

    # 3) Compute per-agent raw metrics and dominant model
    results_rows = []
    summarizer = PromptSummarizer()

    # To later compute scores we need messages/unique_users/diversity for all agents
    metrics_store = {
        "id": [],
        "messages": [],
        "unique_users": [],
        "hhi": [],
        "diversity": [],
    }

    # Also keep dominant model for each agent
    dominant_model_map: Dict[str, Optional[str]] = {}

    # We match per agent
    for idx, ag in df_agents.iterrows():
        rows = match_rows_for_agent(df_logs, ag)
        messages, unique_users, hhi, diversity, dominant_model = compute_agent_metrics(rows)
        metrics_store["id"].append(ag["id"])
        metrics_store["messages"].append(messages)
        metrics_store["unique_users"].append(unique_users)
        metrics_store["hhi"].append(hhi)
        metrics_store["diversity"].append(diversity)
        dominant_model_map[ag["id"]] = dominant_model

    df_metrics = pd.DataFrame(metrics_store).set_index("id")

    # 4) Score computation scope
    if args.score_scope == "per-folder":
        # Compute score separately for each visibility (but do NOT add a duplicate 'visibility' column here)
        df_scores_list = []
        for vis, sub_ids in df_agents.groupby("visibility")["id"]:
            sub_metrics = df_metrics.loc[sub_ids.values]
            sub_scores = score_agents(sub_metrics, alpha=args.alpha)
            df_scores_list.append(sub_scores)
        df_scores = pd.concat(df_scores_list, axis=0)
    else:
        # Global scoring across all selected visibilities
        df_scores = score_agents(df_metrics, alpha=args.alpha)

    # 5) Join back static fields from YAML and dominant model, create final rows
    df_join = df_agents.set_index("id").join(df_scores, how="left")

    df_join["model"] = df_join.index.map(dominant_model_map.get)

    # Summaries from prompts
    df_join["description"] = df_join["prompt"].apply(lambda p: summarizer.summarize(str(p)))

    # 6) Emit per-visibility JSONs
    diagnostics = {
        "drops": drop_stats,
        "n_agents": int(len(df_agents)),
        "n_logs": int(len(df_logs)),
        "score_scope": args.score_scope,
    }

    for vis in visibilities:
        sub = df_join[df_join["visibility"] == vis].copy()
        # Ensure NaN/inf are serialized as null
        records = []
        for _, row in sub.iterrows():
            rec = {
                "description": row.get("description"),
                "id": row.name,
                "name": row.get("name"),
                "owner": row.get("owner"),
                "score": None if pd.isna(row.get("score")) else float(row.get("score")),
                "visibility": row.get("visibility"),
                "model": row.get("model"),
                "prompt": row.get("prompt"),
                "temperature": None if pd.isna(row.get("temperature")) else float(row.get("temperature")),
            }
            # Clean infinities/NaNs
            for k in ["score", "temperature"]:
                v = rec[k]
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
            records.append(rec)

        out_path = Path(args.out_dir) / f"agents_enriched_{vis}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        if args.verbose:
            print(f"[WRITE] {out_path} → {len(records)} agents")

    # 7) Diagnostics
    diag_path = Path(args.out_dir) / "agents_enriched_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    if args.verbose:
        print(f"[WRITE] {diag_path}")


if __name__ == "__main__":
    main()
