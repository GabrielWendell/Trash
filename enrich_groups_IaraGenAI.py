#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA Groups → Enriched JSON per group (PURE IaraGenAI descriptions)
==================================================================

This script scans `groups/`, cross-matches group agents with EVA logs to
compute engagement metrics and a recommendation score, and writes one JSON per
*group* with the following fields per agent:

    description, id, name, owner, score, shared_with, visibility, model, prompt, temperature

Key points
---------
- **No heuristics at all** for descriptions. Text is generated **exclusively**
  by Itaú's IaraGenAI (`mrm_copilot.core.agents.Agent`). If Iara fails/returns
  empty, a placeholder string is saved (no regex/template fallback).
- Compatible with `--score-scope per-group` **or** `--score-scope global`.
  For `global`, scores are computed over **all groups together** using a
  MultiIndex `(group_id, agent_id)` to avoid collisions.
- Filters EVA logs per your governance rules: removes `page == landing`, and
  rows with empty `selected_agent` or `model`.

Example
-------
python enrich_groups_IaraGenAI.py \
  --groups-root "Agents_Chatbots/s3_agents_download/groups" \
  --logs-dir logs_csv/with_model_column \
  --out-dir results_groups \
  --alpha 0.15 --decay 0.0 --score-scope global \
  --iara-model gpt-4.1-2025-04-14 --iara-retries 2 --iara-sleep 0.6 \
  --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ----------------------------- CLI ------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich EVA group agents with IaraGenAI-only descriptions")
    p.add_argument("--groups-root", required=True, type=str,
                   help="Path to groups root: .../s3_agents_download/groups")
    p.add_argument("--logs-dir", required=True, type=str, help="Directory with CSV logs")
    p.add_argument("--out-dir", required=True, type=str, help="Directory to write JSON outputs")
    p.add_argument("--alpha", type=float, default=0.15, help="Diversity penalty floor [0..1]")
    p.add_argument("--decay", type=float, default=0.0, help="Recency exponential decay per day (0 disables)")
    p.add_argument("--score-scope", choices=["per-group", "global"], default="per-group",
                   help="Score normalization within each group or globally across all groups")
    # Iara controls
    p.add_argument("--iara-model", type=str, default="gpt-4.1-2025-04-14", help="IaraGenAI model name")
    p.add_argument("--iara-retries", type=int, default=2, help="Retries per description on transient errors")
    p.add_argument("--iara-sleep", type=float, default=0.6, help="Seconds to sleep between Iara calls")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ----------------------- Log loading & filters -------------------------

REQUIRED_LOG_COLUMNS = [
    "timestamp","user","page","message","__line__","type","selected_agent","model"
]
TIMESTAMP_FMT = "%Y-%m-%d-%H-%M-%S"


def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, engine="python", dtype=str, keep_default_na=True,
                         na_values=["", "None", "null", "NaN", "nan"], on_bad_lines="skip")
    except TypeError:  # pandas < 2 fallback
        df = pd.read_csv(path, engine="python", dtype=str, keep_default_na=True,
                         na_values=["", "None", "null", "NaN", "nan"], error_bad_lines=False, warn_bad_lines=True)
    for c in REQUIRED_LOG_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[REQUIRED_LOG_COLUMNS]


def parse_timestamp_col(df: pd.DataFrame) -> pd.Series:
    def _one(x: str) -> pd.Timestamp:
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
    return df["timestamp"].apply(_one)


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

# -------------------------- Scoring utils -----------------------------

def log_normalize(s: pd.Series) -> pd.Series:
    maxv = float(s.max()) if len(s) else 0.0
    if maxv <= 0.0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return np.log1p(s.astype(float)) / math.log1p(maxv)

# -------------------------- Iara-only generator -----------------------

class IaraOnlyDescriptionGenerator:
    """Itaú IaraGenAI wrapper (no heuristics, no fallback).

    - Fixed **Portuguese** system prompt with the required template.
    - Sends agent YAML `prompt` as `content`.
    - Retries on transient errors and sleeps between calls.
    - If it still fails, returns a placeholder string.
    """
    def __init__(self, model: str, retries: int = 2, sleep_s: float = 0.6, verbose: bool=False):
        self.model = model
        self.retries = max(0, int(retries))
        self.sleep_s = max(0.0, float(sleep_s))
        self.verbose = verbose
        try:
            from mrm_copilot.core.agents import Agent  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Falha ao importar IaraGenAI Agent: {e}")
        system_prompt = (
            "Gere UMA única frase em português (<= 240 caracteres), seguindo o modelo: "
            "'Este agente é especializado em [resumo da tarefa]. Deve ser usado quando você precisar [objetivo do usuário], "
            "atuando como [persona]. Use este agente para [como/quando usar, com base nas entradas do usuário].' "
            "Reformule com suas próprias palavras; NÃO copie trechos do prompt. Mantenha tom profissional."
        )
        try:
            self.agent = Agent(prompt=system_prompt, temperature=0, client="IaraGenAi", model=self.model)
        except Exception as e:
            raise RuntimeError(f"Falha ao instanciar IaraGenAI Agent: {e}")

    def summarize(self, agent_prompt: str) -> str:
        text = (agent_prompt or "").strip()
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                ans = self.agent.interaction(content=text, memory=False, return_type="string")
                if self.sleep_s:
                    time.sleep(self.sleep_s)
                ans = (ans or "").strip()
                if ans:
                    return ans
            except Exception as e:
                last_err = e
                if self.verbose:
                    print(f"[IARA] tentativa {attempt+1} falhou: {e}")
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        return "Descrição indisponível (falha na geração pelo Iara)."

# --------------------------- Core pipeline ----------------------------

def _slug(s: str) -> str:
    return (s or "").replace("/", "_").replace("\\", "_").replace(" ", "_")


def compute_group_metrics(DF: pd.DataFrame, DF_AG: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """Return (metrics_df indexed by id, dominant_model_map)."""
    metr_rows: List[Dict[str, object]] = []
    dom_model: Dict[str, Optional[str]] = {}
    for _, ag in DF_AG.iterrows():
        ag_id = str(ag["id"]) ; ag_nm = str(ag.get("name", ""))
        # Prefer id; fallback to exact name match
        mask = DF["selected_agent"].str.casefold() == ag_id.casefold()
        rows_df = DF.loc[mask]
        if len(rows_df) == 0 and ag_nm:
            rows_df = DF.loc[DF["selected_agent"].str.casefold() == ag_nm.casefold()]
        if len(rows_df) == 0:
            messages = 0.0; unique_users = 0; hhi = 1.0; diversity = 0.0; dominant_model = None
        else:
            messages = float(rows_df["w"].sum())
            unique_users = int(rows_df["user"].nunique())
            per_user_w = rows_df.groupby("user")["w"].sum()
            tot = float(per_user_w.sum())
            if tot <= 0.0:
                hhi = 1.0
            else:
                shares = (per_user_w / tot).astype(float)
                hhi = float((shares**2).sum())
            diversity = max(0.0, min(1.0, 1.0 - hhi))
            # dominant model
            counts = rows_df.groupby("model").size().sort_values(ascending=False)
            if len(counts) == 0:
                dominant_model = None
            else:
                topc = counts.iloc[0]
                tops = counts[counts == topc].index.tolist()
                if len(tops) == 1:
                    dominant_model = str(tops[0])
                else:
                    latest = rows_df.groupby("model")["timestamp"].max()
                    dominant_model = str(latest.loc[tops].idxmax())
        metr_rows.append({"id": ag_id, "messages": messages, "unique_users": unique_users, "hhi": hhi, "diversity": diversity})
        dom_model[ag_id] = dominant_model
    DF_METR = (pd.DataFrame(metr_rows).set_index("id") if metr_rows else
               pd.DataFrame({
                   "id": DF_AG["id"],
                   "messages": np.zeros(len(DF_AG)),
                   "unique_users": np.zeros(len(DF_AG), dtype=int),
                   "hhi": np.ones(len(DF_AG)),
                   "diversity": np.zeros(len(DF_AG)),
               }).set_index("id"))
    return DF_METR, dom_model

# ------------------------------- Main ---------------------------------

def main() -> None:
    args = parse_args()
    groups_root = Path(args.groups_root)
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load logs once
    csvs = sorted(logs_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV logs in {logs_dir}")
    DF = pd.concat([read_csv_robust(p) for p in csvs], ignore_index=True)
    DF["timestamp"] = parse_timestamp_col(DF)
    DF, drop_stats = policy_filter(DF, verbose=args.verbose)
    DF = standardize_strings(DF)
    bad_ts = DF["timestamp"].isna()
    if bad_ts.any():
        drop_stats["dropped_invalid_timestamp"] = int(bad_ts.sum())
        DF = DF.loc[~bad_ts]
    DF = add_recency_weight(DF, args.decay)
    if args.verbose:
        print(f"[LOGS] Rows after filters: {len(DF)} | users={DF['user'].nunique()} | agents={DF['selected_agent'].nunique()} | models={DF['model'].nunique()}")

    # Iara init (pure LLM, no heuristics)
    iara = IaraOnlyDescriptionGenerator(
        model=args.iara_model,
        retries=args.iara_retries,
        sleep_s=args.iara_sleep,
        verbose=args.verbose,
    )

    # Accumulators for scoring/writing
    global_metrics_list: List[pd.DataFrame] = []  # for global scope, use MultiIndex (group_id, id)
    per_group_bundle: List[Tuple[str, pd.DataFrame, pd.DataFrame, str, List[str], Dict[str, Optional[str]]]] = []

    # Iterate groups
    for group_dir in sorted([d for d in groups_root.iterdir() if d.is_dir()]):
        group_id = group_dir.name
        group_yaml = group_dir / f"{group_id}.yaml"
        ag_dir = group_dir / "group_agents"
        if not group_yaml.exists():
            if args.verbose:
                print(f"[WARN] Missing group YAML: {group_yaml}")
            continue
        G = safe_read_yaml(group_yaml)
        # agents listed in the group YAML (dedup preserve order)
        given_ids = [str(a).strip() for a in G.get("agents", []) if str(a).strip()]
        seen = set(); order_ids: List[str] = []
        for a in given_ids:
            if a not in seen:
                seen.add(a); order_ids.append(a)
        members = [normalize_email(m) for m in G.get("members", G.get("membros", [])) if m]
        owner = normalize_email(G.get("owner", ""))

        # Build DF_AG from YAMLs under group_agents
        rows: List[Dict[str, object]] = []
        if ag_dir.exists():
            for yml in sorted(ag_dir.glob("*.yaml")):
                Y = safe_read_yaml(yml)
                ag_id = str((Y.get("agent_id") or Y.get("id") or Y.get("id_agente") or yml.stem)).strip()
                ag_nm = str((Y.get("agent_name") or Y.get("nome_agente") or Y.get("name") or "")).strip()
                pr    = str(Y.get("prompt", ""))
                tmp   = Y.get("temp", Y.get("temperature", None))
                try:
                    tmp = float(tmp) if tmp is not None else None
                except Exception:
                    tmp = None
                rows.append({"id": ag_id, "name": normalize_name(ag_nm), "prompt": pr, "temperature": tmp})
        DF_AG = pd.DataFrame(rows)

        # Ensure IDs listed in the group YAML exist in DF_AG
        existing_ids = set(DF_AG["id"]) if not DF_AG.empty else set()
        for mid in order_ids:
            if mid not in existing_ids:
                DF_AG = pd.concat([DF_AG, pd.DataFrame([{ "id": mid, "name": "", "prompt": "", "temperature": None }])], ignore_index=True)
                existing_ids.add(mid)

        # If group has no agents after all, write empty JSON and continue
        if DF_AG.empty:
            out_path = out_dir / f"groups_enriched_{_slug(group_id)}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            if args.verbose:
                print(f"[WRITE] {out_path} → 0 agents (no YAMLs and no agent IDs listed)")
            # diagnostics
            with open(out_dir / "groups_enriched_diagnostics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"group_id": group_id, "n_agents": 0}, ensure_ascii=False) + "\n")
            continue

        # Compute metrics + dominant model for the group
        DF_METR, dom_model = compute_group_metrics(DF, DF_AG)

        if args.score_scope == "global":
            tmp = DF_METR.copy()
            tmp.index.name = "id"
            tmp["__group_id__"] = group_id
            tmp = tmp.set_index("__group_id__", append=True)  # MultiIndex: (id, group_id)
            tmp = tmp.reorder_levels(["__group_id__", "id"])  # (group_id, id)
            global_metrics_list.append(tmp)
        per_group_bundle.append((group_id, DF_AG, DF_METR, owner, members, dom_model))

    # Scoring
    def score_frame(dfm: pd.DataFrame) -> pd.DataFrame:
        na = log_normalize(dfm["messages"]) ; nu = log_normalize(dfm["unique_users"]) ; div = dfm["diversity"].clip(0,1)
        scoreH = (2.0*na*nu)/(na+nu) ; scoreH[(na+nu)==0.0] = 0.0
        score = scoreH * (args.alpha + (1.0 - args.alpha) * div)
        return dfm.assign(scoreH=scoreH, score=score)

    global_scores: Optional[pd.DataFrame] = None
    if args.score_scope == "global" and global_metrics_list:
        GALL = pd.concat(global_metrics_list, axis=0)  # index: (group_id, id)
        global_scores = score_frame(GALL)

    # Prepare diagnostics file (truncate)
    diag_path = out_dir / "groups_enriched_diagnostics.jsonl"
    with open(diag_path, "w", encoding="utf-8") as _:
        pass

    # Emit outputs per group
    for group_id, DF_AG, DF_METR, owner, members, dom_model in per_group_bundle:
        if args.score_scope == "per-group":
            DF_SCO = score_frame(DF_METR)
        else:
            assert global_scores is not None
            key = (group_id,)
            # slice MultiIndex by first level (group_id)
            DF_SCO = global_scores.loc[key]
            if not isinstance(DF_SCO, pd.DataFrame):
                DF_SCO = global_scores.loc[[key]]  # ensure DataFrame
            # Ensure index is just agent id for downstream lookups
            DF_SCO.index.name = "id"

        records: List[Dict[str, object]] = []
        for _, ag in DF_AG.iterrows():
            ag_id = str(ag["id"]) ; ag_nm = ag.get("name") or ""
            # Description via Iara (no heuristic)
            desc = iara.summarize(str(ag.get("prompt", "")))
            score_val = 0.0
            if ag_id in DF_SCO.index:
                val = DF_SCO.loc[ag_id, "score"]
                score_val = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
            rec = {
                "description":  desc,
                "id":           ag_id,
                "name":         ag_nm or None,
                "owner":        owner or None,
                "score":        score_val,
                "shared_with":  members or [],
                "visibility":   "group",
                "model":        dom_model.get(ag_id),
                "prompt":       ag.get("prompt", ""),
                "temperature":  (None if pd.isna(ag.get("temperature")) else float(ag.get("temperature"))) if ("temperature" in ag) else None,
            }
            # sanitize floats
            for k in ("score", "temperature"):
                v = rec[k]
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
            records.append(rec)

        out_path = out_dir / f"groups_enriched_{_slug(group_id)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        if args.verbose:
            print(f"[WRITE] {out_path} → {len(records)} agents")

        with open(diag_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"group_id": group_id, "n_agents": len(records)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
