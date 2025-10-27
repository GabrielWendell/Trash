#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA Agents → Enriched JSON (Public/Private) with `username` and structured descriptions
=====================================================================================

This version keeps the same behavior as `enrich_agents_username.py` but changes
how the description is created from the agent prompt. Instead of a generic
summary, it generates text following this template:

  "This agent specializes in [task summary]. It will be used when you need
   [user goal], acting as [persona]. Use this agent to [brief explanation of how
   and when to use it, based on user responses]."

The extraction is rule‑based (no external LLM call): it searches the prompt for
common cue words ("you are", "act as", "role:", "goal:", "objective:",
"help/assist to ...") and falls back gracefully when cues are missing.

CLI (example)
-------------
python enrich_agents_username_structdesc.py \
  --agents-root "ChatBots Files/s3_agents_download" \
  --logs-dir logs_csv \
  --out-dir results_enriched \
  --visibilities public private \
  --alpha 0.15 \
  --decay 0.0 \
  --score-scope global \
  --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
    p = argparse.ArgumentParser(description="Enrich EVA agents by cross-matching YAMLs with logs (username + structured description)")
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
    for col in REQUIRED_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[REQUIRED_LOG_COLUMNS]


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
    return " ".join((s or "").split()).strip()


def email_to_username(email: str) -> str:
    if not email:
        return ""
    local = str(email).split("@", 1)[0]
    for sep in [".", "-", "_", "+"]:
        local = local.replace(sep, " ")
    tokens = [t for t in local.split() if t]
    def norm(tok: str) -> str:
        if len(tok) == 1:
            return tok.upper()
        return tok[0].upper() + tok[1:].lower()
    return " ".join(norm(t) for t in tokens)


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
    big = pd.concat([read_csv_robust(p) for p in csvs], ignore_index=True)

    big["timestamp"] = parse_timestamp_col(big)
    big, drop_stats = policy_filter(big, verbose=verbose)
    big = standardize_strings(big)

    bad_ts = big["timestamp"].isna()
    if bad_ts.any():
        drop_stats["dropped_invalid_timestamp"] = int(bad_ts.sum())
        big = big.loc[~bad_ts]

    big = add_recency_weight(big, decay)

    if verbose:
        print(f"[LOGS] Rows after filters: {len(big)} | users={big['user'].nunique()} | agents={big['selected_agent'].nunique()} | models={big['model'].nunique()}")
    return big, drop_stats

# --------------------------------------------------------------------------------------
# Matching & Metrics
# --------------------------------------------------------------------------------------

def match_rows_for_agent(df_logs: pd.DataFrame, ag: pd.Series) -> pd.DataFrame:
    mask_id = df_logs["selected_agent"].str.casefold() == ag["id"].casefold()
    if mask_id.any():
        return df_logs.loc[mask_id]
    mask_nm = df_logs["selected_agent"].str.casefold() == ag["name"].casefold()
    return df_logs.loc[mask_nm]


def compute_agent_metrics(df_rows: pd.DataFrame) -> Tuple[float, int, float, float, Optional[str]]:
    if df_rows is None or len(df_rows) == 0:
        return 0.0, 0, 1.0, 0.0, None
    messages = float(df_rows["w"].sum())
    unique_users = int(df_rows["user"].nunique())
    per_user_w = df_rows.groupby("user")["w"].sum().rename("wcount")
    total = float(per_user_w.sum())
    if total <= 0.0:
        hhi = 1.0
    else:
        shares = (per_user_w / total).astype(float)
        hhi = float((shares ** 2).sum())
    diversity = max(0.0, min(1.0, 1.0 - hhi))
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
# Structured prompt → description
# --------------------------------------------------------------------------------------

class PromptSummarizerStructured:
    """Create a short, structured description from a free-text prompt.

    Template:
        This agent specializes in [task]. It will be used when you need
        [goal], acting as [persona]. Use this agent to [how/when].
    """
    RE_ROLE = re.compile(r"\b(?:you are|you're|act as|role\s*[:=])\s*(.+?)(?:[\.;\n]|$)", re.IGNORECASE)
    RE_GOAL = re.compile(r"\b(?:goal|objective|purpose|mission|task)\s*[:=]\s*(.+?)(?:[\.;\n]|$)", re.IGNORECASE)
    RE_HELP = re.compile(r"\b(?:help|assist|support)\b(?:[^\n\.;]{0,40})\b(?:to|in|with)\s+(.+?)(?:[\.;\n]|$)", re.IGNORECASE)

    def __init__(self):
        self.cache: Dict[str, str] = {}

    @staticmethod
    def _clean(s: str, max_len: int = 140) -> str:
        s = " ".join(s.split())
        if len(s) <= max_len:
            return s
        # try to trim at punctuation
        cut = s[:max_len]
        for ch in ".;,: ":
            pos = cut.rfind(ch)
            if pos >= 60:
                return cut[:pos].strip()
        return cut.strip()

    def _extract(self, prompt: str) -> Tuple[str, str, str, str]:
        text = (prompt or "").strip()
        if not text:
            return (
                "solving the tasks described in its prompt",
                "to achieve results aligned with those objectives",
                "a helpful assistant",
                "ask focused questions and provide the relevant context or data",
            )
        # persona / role
        m_role = self.RE_ROLE.search(text)
        persona = self._clean(m_role.group(1)) if m_role else "a helpful assistant"
        # goal (explicit)
        m_goal = self.RE_GOAL.search(text)
        goal = self._clean(m_goal.group(1)) if m_goal else "reach your desired outcome"
        # task (implicit via first sentence or help pattern)
        m_help = self.RE_HELP.search(text)
        if m_help:
            task = self._clean(m_help.group(1))
        else:
            # fallback: use first sentence fragment as task summary
            first = re.split(r"[\n\.!?]", text, maxsplit=1)[0]
            task = self._clean(first) if first else "the tasks described in its prompt"
        # how/when to use (combine light guidance)
        how = "provide inputs, context, and any constraints to obtain precise, actionable outputs"
        return (task, goal, persona, how)

    def summarize(self, prompt: str) -> str:
        key = (prompt or "").strip()
        if key in self.cache:
            return self.cache[key]
        task, goal, persona, how = self._extract(key)
        desc = (
            f"This agent specializes in {task}. "
            f"It will be used when you need {goal}, acting as {persona}. "
            f"Use this agent to {how}."
        )
        self.cache[key] = desc
        return desc

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
    df_agents["username"] = df_agents["owner"].apply(email_to_username)

    # 2) Load logs and prepare
    df_logs, drop_stats = load_and_prepare_logs(logs_dir, decay=args.decay, verbose=args.verbose)

    # 3) Compute per-agent metrics and dominant model
    metrics_store = {"id": [], "messages": [], "unique_users": [], "hhi": [], "diversity": []}
    dominant_model_map: Dict[str, Optional[str]] = {}

    for _, ag in df_agents.iterrows():
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
        df_scores_list = []
        for _, sub_ids in df_agents.groupby("visibility")["id"]:
            sub_metrics = df_metrics.loc[sub_ids.values]
            sub_scores = score_agents(sub_metrics, alpha=args.alpha)
            df_scores_list.append(sub_scores)
        df_scores = pd.concat(df_scores_list, axis=0)
    else:
        df_scores = score_agents(df_metrics, alpha=args.alpha)

    # 5) Join static fields and dominant model
    df_join = df_agents.set_index("id").join(df_scores, how="left")
    df_join["model"] = df_join.index.map(dominant_model_map.get)

    # Structured summaries from prompts
    summarizer = PromptSummarizerStructured()
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
        records = []
        for _, row in sub.iterrows():
            rec = {
                "description": row.get("description"),
                "id": row.name,
                "name": row.get("name"),
                "owner": row.get("owner"),
                "username": row.get("username"),
                "score": None if pd.isna(row.get("score")) else float(row.get("score")),
                "visibility": row.get("visibility"),
                "model": row.get("model"),
                "prompt": row.get("prompt"),
                "temperature": None if pd.isna(row.get("temperature")) else float(row.get("temperature")),
            }
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
