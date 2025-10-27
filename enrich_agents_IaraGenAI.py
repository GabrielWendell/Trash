#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA Agents → Enriched JSON (Public/Private) with usernames + IaraGenAI descriptions
=================================================================================

This script scans agents in `public/` and `private/`, cross-matches with EVA
logs to compute engagement metrics and a recommendation score, and emits
per-visibility JSON files with the following fields per agent:

    description, id, name, owner, username, score, visibility, model,
    prompt, temperature

**New**: Uses Itaú's internal LLM (IaraGenAI) via `mrm_copilot.core.agents.Agent`
with a fixed system prompt to generate the description from each agent's YAML
prompt. If Iara is unavailable or fails, we fall back to a robust heuristic
summarizer (no external calls).

Example:
    python enrich_agents_username.py \
      --agents-root "ChatBots Files/s3_agents_download" \
      --logs-dir logs_csv \
      --out-dir results_enriched \
      --visibilities public private \
      --alpha 0.15 --decay 0.0 --score-scope global \
      --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ----------------------------- CLI ------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich EVA agents by cross-matching YAMLs with logs (IaraGenAI descriptions)")
    p.add_argument("--agents-root", required=True, type=str,
                   help="Root path containing public/ and private/ subfolders")
    p.add_argument("--logs-dir", required=True, type=str,
                   help="Directory with pre-converted CSV logs (with model column)")
    p.add_argument("--out-dir", required=True, type=str, help="Directory to write JSON outputs")
    p.add_argument("--visibilities", nargs="*", default=["public", "private"],
                   help="Which visibilities to include (default: public private)")
    p.add_argument("--alpha", type=float, default=0.15, help="Diversity penalty floor [0..1]")
    p.add_argument("--decay", type=float, default=0.0, help="Recency exponential decay per day (0 disables)")
    p.add_argument("--score-scope", choices=["global", "per-folder"], default="global",
                   help="Score normalization across all selected folders or per folder")
    p.add_argument("--use-iara", action="store_true", help="Use IaraGenAI to generate descriptions (fallback to heuristic if it fails)")
    p.add_argument("--iara-model", type=str, default="gpt-4.1-2025-04-14", help="IaraGenAI model name")
    p.add_argument("--iara-retries", type=int, default=2, help="Retries per description when Iara throttles/errors")
    p.add_argument("--iara-sleep", type=float, default=0.6, help="Seconds to sleep between Iara calls to be polite")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ----------------------- Log loading & filters -------------------------

REQUIRED_LOG_COLUMNS = [
    "timestamp", "user", "page", "message", "__line__", "type", "selected_agent", "model"
]
TIMESTAMP_FMT = "%Y-%m-%d-%H-%M-%S"  # e.g., 2025-09-01-08-43-38
VALID_VISIBILITIES = {"public", "private"}


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, engine="python", dtype=str, keep_default_na=True,
                         na_values=["", "None", "null", "NaN", "nan"], on_bad_lines="skip")
    except TypeError:  # pandas<2
        df = pd.read_csv(path, engine="python", dtype=str, keep_default_na=True,
                         na_values=["", "None", "null", "NaN", "nan"], error_bad_lines=False, warn_bad_lines=True)
    for c in REQUIRED_LOG_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[REQUIRED_LOG_COLUMNS]


def parse_timestamp_col(df: pd.DataFrame) -> pd.Series:
    def _parse_one(x: str) -> pd.Timestamp:
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        try:
            return pd.to_datetime(s, format=TIMESTAMP_FMT, utc=True)
        except Exception:
            try:
                return pd.to_datetime(s, utc=True)
            except Exception:
                return pd.NaT
    return df["timestamp"].apply(_parse_one)


def empty_like_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip().str.lower()
    return s.isna() | s_str.isin(["", "nan", "none", "null"])


def policy_filter(df: pd.DataFrame, verbose: bool=False) -> Tuple[pd.DataFrame, Dict[str,int]]:
    stats: Dict[str,int] = {}
    n0 = len(df)
    mask = df["page"].astype(str).str.strip().str.lower() == "landing"
    stats["dropped_landing"] = int(mask.sum()); df = df.loc[~mask]
    mask = empty_like_series(df["selected_agent"]) ; stats["dropped_selected_agent_invalid"] = int(mask.sum()); df = df.loc[~mask]
    mask = empty_like_series(df["model"])        ; stats["dropped_model_invalid"]         = int(mask.sum()); df = df.loc[~mask]
    stats["kept_after_filters"] = int(len(df)); stats["dropped_total"] = int(n0 - len(df))
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
        df["w"] = 1.0; return df
    now = pd.Timestamp.utcnow()
    age_days = (now - df["timestamp"]).dt.total_seconds() / 86400.0
    df["w"] = np.exp(-decay * age_days.astype(float))
    return df

# -------------------------- YAML helpers ------------------------------

@dataclass
class AgentRow:
    id: str
    name: str
    owner: str
    prompt: str
    temperature: float
    visibility: str  # public/private


def safe_read_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
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
    toks = [t for t in local.split() if t]
    def cap(tok: str) -> str:
        return tok.upper() if len(tok) == 1 else tok[0].upper() + tok[1:].lower()
    return " ".join(cap(t) for t in toks)


def scan_agents(root: Path, visibilities: Iterable[str], verbose: bool=False) -> pd.DataFrame:
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
                Y = safe_read_yaml(yml)
                ag_id = (Y.get("agent_id") or Y.get("id") or Y.get("id_agente") or yml.stem or "").strip()
                ag_nm = (Y.get("agent_name") or Y.get("nome_agente") or Y.get("name") or "").strip()
                pr    = Y.get("prompt") or ""
                tmp   = Y.get("temp", Y.get("temperature", 0.0))
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

# ----------------------- Logs load + prep ------------------------------

def load_and_prepare_logs(logs_dir: Path, decay: float, verbose: bool=False) -> Tuple[pd.DataFrame, Dict[str,int]]:
    csvs = sorted(logs_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV logs in {logs_dir}")
    DF = pd.concat([read_csv_robust(p) for p in csvs], ignore_index=True)
    DF["timestamp"] = parse_timestamp_col(DF)
    DF, drop_stats = policy_filter(DF, verbose=verbose)
    DF = standardize_strings(DF)
    bad_ts = DF["timestamp"].isna()
    if bad_ts.any():
        drop_stats["dropped_invalid_timestamp"] = int(bad_ts.sum())
        DF = DF.loc[~bad_ts]
    DF = add_recency_weight(DF, decay)
    if verbose:
        print(f"[LOGS] Rows after filters: {len(DF)} | users={DF['user'].nunique()} | agents={DF['selected_agent'].nunique()} | models={DF['model'].nunique()}")
    return DF, drop_stats

# -------------------------- Metrics & scoring -------------------------

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
    per_user_w = df_rows.groupby("user")["w"].sum()
    tot = float(per_user_w.sum())
    if tot <= 0.0:
        hhi = 1.0
    else:
        shares = (per_user_w / tot).astype(float)
        hhi = float((shares**2).sum())
    diversity = max(0.0, min(1.0, 1.0 - hhi))
    # dominant model
    counts = df_rows.groupby("model").size().sort_values(ascending=False)
    if len(counts) == 0:
        dominant_model = None
    else:
        topc = counts.iloc[0]
        tops = counts[counts == topc].index.tolist()
        if len(tops) == 1:
            dominant_model = str(tops[0])
        else:
            latest = df_rows.groupby("model")["timestamp"].max()
            dominant_model = str(latest.loc[tops].idxmax())
    return messages, unique_users, hhi, diversity, dominant_model


def log_normalize(s: pd.Series) -> pd.Series:
    maxv = float(s.max()) if len(s) else 0.0
    if maxv <= 0.0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return np.log1p(s.astype(float)) / math.log1p(maxv)


def score_agents(df_metrics: pd.DataFrame, alpha: float) -> pd.DataFrame:
    na = log_normalize(df_metrics["messages"])  # [0,1]
    nu = log_normalize(df_metrics["unique_users"])  # [0,1]
    scoreH = (2.0*na*nu)/(na+nu) ; scoreH[(na+nu)==0.0] = 0.0
    div = df_metrics["diversity"].astype(float).clip(0,1)
    score = scoreH * (alpha + (1.0 - alpha) * div)
    return df_metrics.assign(scoreH=scoreH, score=score)

# ---------------- IaraGenAI + heuristic description generator ---------

class HeuristicSummarizer:
    """Pure heuristic template-based description generator (no LLM)."""
    BASE_VERBS = {
        "write": "writing", "generate": "generating", "create": "creating",
        "summarize": "summarizing", "classify": "classifying", "extract": "extracting",
        "analyze": "analyzing", "analyse": "analyzing", "evaluate": "evaluating",
        "compute": "computing", "forecast": "forecasting", "predict": "predicting",
        "plan": "planning", "draft": "drafting", "format": "formatting",
        "translate": "translating", "validate": "validating", "compare": "comparing",
        "document": "documenting", "assist": "assisting", "help": "helping",
        "explain": "explaining", "review": "reviewing", "refactor": "refactoring",
        "debug": "debugging", "query": "querying", "visualize": "visualizing",
    }
    RE_ROLE = re.compile(r"\b(?:you are|you're|act as|role\s*[:=])\s*(.+?)(?:[\.;\n]|$)", re.I)
    RE_GOAL = re.compile(r"\b(?:goal|objective|purpose|mission|task)\s*[:=]\s*(.+?)(?:[\.;\n]|$)", re.I)
    RE_TO_VERB   = re.compile(r"\bto\s+([a-z]{3,})([^\.;\n]*)", re.I)
    RE_WILL_VERB = re.compile(r"\bwill\s+(?:be\s+)?([a-z]{3,})(?:ing\b)?([^\.;\n]*)", re.I)
    RE_VERB_ING  = re.compile(r"\b([a-z]{3,})ing\b([^\.;\n]*)", re.I)

    def _clean(self, s: str) -> str:
        s = " ".join((s or "").split())
        s = re.sub(r"^(?:an?|the)\s+", "", s, flags=re.I)
        return s

    def _to_gerund(self, v: str) -> str:
        v = v.lower()
        return self.BASE_VERBS.get(v, v + ("ing" if not v.endswith("e") else "ing"))

    def _persona(self, prompt: str) -> str:
        m = self.RE_ROLE.search(prompt)
        return self._clean(m.group(1)) if m else "a helpful assistant"

    def _compose(self, verb: str, tail: str) -> str:
        ger = self._to_gerund(verb)
        tail = self._clean(tail)
        tail = re.split(r"\b(?:that|which|who|so that|in order to)\b", tail, 1, flags=re.I)[0]
        words = [w for w in tail.split() if w]
        tail = " ".join(words[:12])
        return (ger + (" " + tail if tail else "")).strip()

    def _tasks(self, prompt: str) -> List[str]:
        tasks: List[str] = []
        used = set()
        for rgx in (self.RE_TO_VERB, self.RE_WILL_VERB, self.RE_VERB_ING):
            for m in rgx.finditer(prompt):
                v = m.group(1)
                if len(v) < 3 or v.lower() in used:
                    continue
                used.add(v.lower())
                tail = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                tasks.append(self._compose(v, tail))
                if len(tasks) >= 2:
                    break
            if len(tasks) >= 2:
                break
        if tasks:
            return tasks
        # fallback: cleaned first sentence, sans "you are/you will"
        first = re.split(r"[\.!?\n]", prompt, maxsplit=1)[0]
        first = re.sub(r"\b(?:you are|you're|you will|your task is|role\s*[:=])\b\s*", "", first, flags=re.I)
        first = self._clean(first)
        return [first] if first else ["executing the tasks described in its prompt"]

    def _goal(self, prompt: str) -> str:
        m = self.RE_GOAL.search(prompt)
        if m:
            return self._clean(m.group(1))
        m = re.search(r"\bso that\s+([^\.;\n]+)", prompt, re.I)
        if m:
            return self._clean(m.group(1))
        m = re.search(r"\bin order to\s+([^\.;\n]+)", prompt, re.I)
        if m:
            return self._clean(m.group(1))
        m = self.RE_TO_VERB.search(prompt)
        if m:
            return self._clean((m.group(1) + " " + (m.group(2) or "")).strip())
        return "reach your desired outcome"

    def _how(self, prompt: str) -> str:
        if re.search(r"\b(minute|minutes|document|report|agenda|memo|note)\b", prompt, re.I):
            return "provide context and any required structure/templates so it can format documents correctly"
        if re.search(r"\b(audit|auditor|compliance|regulator|risk)\b", prompt, re.I):
            return "share case details and criteria so it can draft compliant, reviewable text"
        if re.search(r"\b(sql|query|database|dataset|table|schema|csv|excel)\b", prompt, re.I):
            return "state questions and constraints; include sample rows/schemas if relevant"
        return "provide the necessary context and constraints for precise, actionable outputs"

    def summarize(self, prompt: str) -> str:
        persona = self._persona(prompt)
        task = "; ".join(self._tasks(prompt))
        goal = self._goal(prompt)
        how = self._how(prompt)
        return (
            f"This agent specializes in {task}. "
            f"It should be used when you need {goal}, acting as {persona}. "
            f"Use this agent to {how}."
        )


class IaraDescriptionGenerator:
    """Wrapper around mrm_copilot.core.agents.Agent with retries and fallback."""
    def __init__(self, enabled: bool, model: str, retries: int, sleep_s: float, verbose: bool=False):
        self.enabled = enabled
        self.model = model
        self.retries = max(0, int(retries))
        self.sleep_s = max(0.0, float(sleep_s))
        self.verbose = verbose
        self.agent = None
        self.heur = HeuristicSummarizer()
        if self.enabled:
            try:
                from mrm_copilot.core.agents import Agent  # type: ignore
                system_prompt = (
                    "This agent specializes in [task summary]. It should be used when you need [user goal], "
                    "acting as [persona]. Use this agent to [brief explanation of how and when to use it, based on user responses].\n"
                    "Given the following agent prompt, output exactly ONE sentence following the template above. "
                    "Do NOT copy the original wording; paraphrase into the template, keep <= 220 characters, no labels."
                )
                self.agent = Agent(prompt=system_prompt, temperature=0, client="IaraGenAi", model=self.model)
                self.ok = True
            except Exception as e:
                if self.verbose:
                    print(f"[IARA] Disabled (init error): {e}")
                self.ok = False
        else:
            self.ok = False

    def summarize(self, agent_prompt: str) -> str:
        p = (agent_prompt or "").strip()
        if not self.ok or not self.agent:
            return self.heur.summarize(p)
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                ans = self.agent.interaction(content=p, memory=False, return_type="string")
                ans = (ans or "").strip()
                if ans:
                    return ans
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"[IARA] attempt {attempt+1} failed: {e}")
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        if self.verbose and last_err is not None:
            print(f"[IARA] Falling back to heuristic after error: {last_err}")
        return self.heur.summarize(p)

# ------------------------------- Main ---------------------------------

def main() -> None:
    args = parse_args()
    agents_root = Path(args.agents_root)
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visibilities = [v for v in args.visibilities if v in VALID_VISIBILITIES]
    if not visibilities:
        raise ValueError("No valid visibilities specified (use: public, private)")

    # 1) Scan YAML agents
    df_agents = scan_agents(agents_root, visibilities, verbose=args.verbose)
    if df_agents.empty:
        raise RuntimeError("No agents found under the specified visibilities.")
    df_agents["username"] = df_agents["owner"].apply(email_to_username)

    # 2) Load and prepare logs
    df_logs, drop_stats = load_and_prepare_logs(logs_dir, decay=args.decay, verbose=args.verbose)

    # 3) Compute metrics and dominant model
    metrics_store = {"id": [], "messages": [], "unique_users": [], "hhi": [], "diversity": []}
    dominant_model_map: Dict[str, Optional[str]] = {}

    for _, ag in df_agents.iterrows():
        rows = match_rows_for_agent(df_logs, ag)
        messages, unique_users, hhi, diversity, dom_model = compute_agent_metrics(rows)
        metrics_store["id"].append(ag["id"])
        metrics_store["messages"].append(messages)
        metrics_store["unique_users"].append(unique_users)
        metrics_store["hhi"].append(hhi)
        metrics_store["diversity"].append(diversity)
        dominant_model_map[ag["id"]] = dom_model

    df_metrics = pd.DataFrame(metrics_store).set_index("id")

    # 4) Score computation (global or per-folder)
    if args.score_scope == "per-folder":
        parts = []
        for _, ids in df_agents.groupby("visibility")["id"]:
            parts.append(score_agents(df_metrics.loc[ids.values], alpha=args.alpha))
        df_scores = pd.concat(parts, axis=0)
    else:
        df_scores = score_agents(df_metrics, alpha=args.alpha)

    # 5) Join static fields + dominant model
    df_join = df_agents.set_index("id").join(df_scores, how="left")
    df_join["model"] = df_join.index.map(dominant_model_map.get)

    # 6) Descriptions via IaraGenAI (with heuristic fallback)
    iara = IaraDescriptionGenerator(
        enabled=args.use_iara,
        model=args.iara_model,
        retries=args.iara_retries,
        sleep_s=args.iara_sleep,
        verbose=args.verbose,
    )
    df_join["description"] = df_join["prompt"].apply(lambda p: iara.summarize(str(p)))

    # 7) Emit per-visibility JSONs
    diagnostics = {
        "drops": drop_stats,
        "n_agents": int(len(df_agents)),
        "n_logs": int(len(df_logs)),
        "score_scope": args.score_scope,
        "iara_enabled": bool(args.use_iara),
        "iara_model": args.iara_model if args.use_iara else None,
    }

    for vis in visibilities:
        sub = df_join[df_join["visibility"] == vis].copy()
        records: List[Dict[str, object]] = []
        for _, row in sub.iterrows():
            rec = {
                "description": row.get("description"),
                "id": row.name,
                "name": row.get("name") or None,
                "owner": row.get("owner") or None,
                "username": row.get("username") or None,
                "score": None if pd.isna(row.get("score")) else float(row.get("score")),
                "visibility": row.get("visibility") or None,
                "model": row.get("model") or None,
                "prompt": row.get("prompt") or None,
                "temperature": None if pd.isna(row.get("temperature")) else float(row.get("temperature")),
            }
            # sanitize score/temperature edge cases
            for k in ("score", "temperature"):
                v = rec[k]
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
            records.append(rec)
        out_path = Path(args.out_dir) / f"agents_enriched_{vis}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        if args.verbose:
            print(f"[WRITE] {out_path} → {len(records)} agents")

    # 8) Diagnostics
    diag_path = Path(args.out_dir) / "agents_enriched_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    if args.verbose:
        print(f"[WRITE] {diag_path}")


if __name__ == "__main__":
    main()
