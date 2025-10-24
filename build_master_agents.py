#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA Groups → Enriched JSON per group (FIXED)
-------------------------------------------

Robust implementation that scans the groups tree and, for each group, emits a
JSON with enriched agent fields by cross-matching agent YAMLs with EVA logs.

Key fixes vs previous draft
- Correct use of Pandas string ops (.str.strip()) in filters.
- Proper indentation inside the group loop (no stray top-level blocks).
- Safe handling of groups with zero agents (writes empty JSON + diagnostics).
- Diagnostics newline bug fixed.
- Carries owner/members and dominant model per group through to the writer.

CLI example
    python enrich_groups_fixed.py \
      --groups-root "Agents_Chatbots/s3_agents_download/groups" \
      --logs-dir logs_csv/with_model_column \
      --out-dir results_groups \
      --alpha 0.15 \
      --decay 0.0 \
      --score-scope per-group \
      --verbose
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ----------------------------- CLI ------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich EVA group agents by cross-matching YAMLs with logs")
    p.add_argument("--groups-root", required=True, type=str,
                   help="Path to groups root: .../s3_agents_download/groups")
    p.add_argument("--logs-dir", required=True, type=str, help="Directory with CSV logs")
    p.add_argument("--out-dir", required=True, type=str, help="Directory to write JSON outputs")
    p.add_argument("--alpha", type=float, default=0.15, help="Diversity penalty floor [0..1]")
    p.add_argument("--decay", type=float, default=0.0, help="Recency exponential decay per day (0 disables)")
    p.add_argument("--score-scope", choices=["per-group", "global"], default="per-group",
                   help="Score normalization within each group or globally across all groups")
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
    except TypeError:
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
    # correct vectorized ops
    s_str = s.astype(str).str.strip().str.lower()
    return s.isna() | s_str.isin(["", "nan", "none", "null"])


def policy_filter(df: pd.DataFrame, verbose: bool=False) -> Tuple[pd.DataFrame, Dict[str,int]]:
    stats: Dict[str,int] = {}
    n0 = len(df)
    # page != landing
    mask = df["page"].astype(str).str.strip().str.lower() == "landing"
    stats["dropped_landing"] = int(mask.sum()); df = df.loc[~mask]
    # valid selected_agent
    mask = empty_like_series(df["selected_agent"]) ; stats["dropped_selected_agent_invalid"] = int(mask.sum()); df = df.loc[~mask]
    # valid model
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

# -------------------- Prompt summarizer (stub) ------------------------

class PromptSummarizer:
    def __init__(self):
        self.cache: Dict[str,str] = {}
    def summarize(self, prompt: str) -> str:
        key = (prompt or "").strip()
        if key in self.cache:
            return self.cache[key]
        if not key:
            desc = "Agent without a defined prompt."
        else:
            desc = self._fallback_summary(key)
        self.cache[key] = desc
        return desc
    @staticmethod
    def _fallback_summary(prompt: str, max_chars: int = 160) -> str:
        t = " ".join(prompt.split())
        if len(t) <= max_chars:
            return t
        cut = t[:max_chars]
        last_dot = cut.rfind(".")
        return cut[:last_dot+1] if last_dot >= 40 else cut + "…"

# -------------------------- Scoring ----------------------------------

def log_normalize(s: pd.Series) -> pd.Series:
    maxv = float(s.max()) if len(s) else 0.0
    if maxv <= 0.0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return np.log1p(s.astype(float)) / math.log1p(maxv)


def harmonic_mean(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = a + b
    hm = (2.0*a*b) / denom
    hm[denom == 0.0] = 0.0
    return hm

# --------------------------- Main ------------------------------------

def _slug(s: str) -> str:
    return (s or "").replace("/", "_").replace("\\", "_").replace(" ", "_")


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
    dfs = [read_csv_robust(p) for p in csvs]
    DF = pd.concat(dfs, ignore_index=True)
    DF["timestamp"] = parse_timestamp_col(DF)
    DF, drop_stats = policy_filter(DF, verbose=args.verbose)
    DF = standardize_strings(DF)
    bad_ts = DF["timestamp"].isna()
    if bad_ts.any():
        DF = DF.loc[~bad_ts]
    DF = add_recency_weight(DF, args.decay)
    if args.verbose:
        print(f"[LOGS] Rows after filters: {len(DF)} | users={DF['user'].nunique()} | agents={DF['selected_agent'].nunique()} | models={DF['model'].nunique()}")

    summarizer = PromptSummarizer()

    # Accumulators for scoring/writing
    global_metrics_frames: List[pd.DataFrame] = []
    # (group_id, DF_AG, DF_METR, owner, members, dom_model_map)
    per_group_outputs: List[Tuple[str, pd.DataFrame, pd.DataFrame, str, List[str], Dict[str, Optional[str]]]] = []

    # Iterate groups
    for group_dir in sorted([d for d in groups_root.iterdir() if d.is_dir()]):
        group_id = group_dir.name
        group_yaml = group_dir / f"{group_id}.yaml"
        group_agents_dir = group_dir / "group_agents"
        if not group_yaml.exists():
            if args.verbose:
                print(f"[WARN] Missing group YAML: {group_yaml}")
            continue
        G = safe_read_yaml(group_yaml)
        agents_list = G.get("agents", [])
        agents_list = [str(a).strip() for a in agents_list if str(a).strip()]
        # dedupe (preserve order)
        seen = set(); dedup_agents: List[str] = []
        for a in agents_list:
            if a not in seen:
                seen.add(a); dedup_agents.append(a)
        members = G.get("members", G.get("membros", []))
        members = [normalize_email(m) for m in members if m]
        owner = normalize_email(G.get("owner", ""))

        # Build DF_AG from YAMLs under group_agents
        rows: List[Dict[str, object]] = []
        if group_agents_dir.exists():
            for yml in sorted(group_agents_dir.glob("*.yaml")):
                Y = safe_read_yaml(yml)
                ag_id = (Y.get("agent_id") or Y.get("id") or Y.get("id_agente") or yml.stem)
                ag_id = str(ag_id).strip()
                ag_nm = (Y.get("agent_name") or Y.get("nome_agente") or Y.get("name") or "")
                ag_nm = str(ag_nm).strip()
                pr    = str(Y.get("prompt", ""))
                tmp   = Y.get("temp", Y.get("temperature", None))
                try:
                    tmp = float(tmp) if tmp is not None else None
                except Exception:
                    tmp = None
                rows.append({"id": ag_id, "name": normalize_name(ag_nm), "prompt": pr, "temperature": tmp})
        DF_AG = pd.DataFrame(rows)

        # Ensure IDs from the group list are represented
        existing_ids = set(DF_AG["id"]) if not DF_AG.empty else set()
        for mid in dedup_agents:
            if mid not in existing_ids:
                DF_AG = pd.concat([DF_AG, pd.DataFrame([{ "id": mid, "name": "", "prompt": "", "temperature": None }])], ignore_index=True)
                existing_ids.add(mid)

        # If still empty → write empty JSON and continue
        if DF_AG.empty:
            out_path = out_dir / f"groups_enriched_{_slug(group_id)}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            if args.verbose:
                print(f"[WRITE] {out_path} → 0 agents (no YAMLs and no agent IDs listed)")
            # diagnostics
            diagnostics_path = out_dir / "groups_enriched_diagnostics.jsonl"
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diagnostics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"group_id": group_id, "n_agents": 0}, ensure_ascii=False) + "\n")
            continue

        # Match logs and compute metrics + dominant model per agent
        metr_rows: List[Dict[str, object]] = []
        dom_model: Dict[str, Optional[str]] = {}
        for _, ag in DF_AG.iterrows():
            ag_id = str(ag["id"]) ; ag_nm = str(ag.get("name", ""))
            mask_id = DF["selected_agent"].str.casefold() == ag_id.casefold()
            if mask_id.any():
                rows_df = DF.loc[mask_id]
            else:
                if ag_nm:
                    mask_nm = DF["selected_agent"].str.casefold() == ag_nm.casefold()
                    rows_df = DF.loc[mask_nm]
                else:
                    rows_df = DF.iloc[0:0]

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

        # Build metrics frame (or zero frame if somehow empty)
        if len(metr_rows) == 0:
            DF_METR = pd.DataFrame({
                "id": DF_AG["id"],
                "messages": np.zeros(len(DF_AG)),
                "unique_users": np.zeros(len(DF_AG), dtype=int),
                "hhi": np.ones(len(DF_AG)),
                "diversity": np.zeros(len(DF_AG)),
            }).set_index("id")
        else:
            DF_METR = pd.DataFrame(metr_rows).set_index("id")

        if args.score_scope == "global":
            global_metrics_frames.append(DF_METR.assign(__group_id__=group_id))
        per_group_outputs.append((group_id, DF_AG, DF_METR, owner, members, dom_model))

    # ---------- Scoring utilities ----------
    def score_frame(dfm: pd.DataFrame) -> pd.DataFrame:
        na = log_normalize(dfm["messages"]) ; nu = log_normalize(dfm["unique_users"]) ; div = dfm["diversity"].clip(0,1)
        scoreH = (2.0*na*nu)/(na+nu) ; scoreH[(na+nu)==0.0] = 0.0
        score = scoreH * (args.alpha + (1.0 - args.alpha) * div)
        return dfm.assign(scoreH=scoreH, score=score)

    global_scores: Optional[pd.DataFrame] = None
    if args.score_scope == "global" and global_metrics_frames:
        GALL = pd.concat(global_metrics_frames, axis=0)
        global_scores = score_frame(GALL.drop(columns=["__group_id__"]))
        global_scores["__group_id__"] = GALL["__group_id__"].values

    # Prepare diagnostics file (truncate)
    diagnostics_path = out_dir / "groups_enriched_diagnostics.jsonl"
    with open(diagnostics_path, "w", encoding="utf-8") as _:
        pass

    # ---------- Write outputs per group ----------
    for group_id, DF_AG, DF_METR, owner, members, dom_model in per_group_outputs:
        if args.score_scope == "per-group":
            DF_SCO = score_frame(DF_METR)
        else:
            assert global_scores is not None
            DF_SCO = global_scores.loc[DF_METR.index]

        records: List[Dict[str, object]] = []
        for _, ag in DF_AG.iterrows():
            ag_id = str(ag["id"])
            desc = summarizer.summarize(str(ag.get("prompt", "")))
            rec = {
                "description":  desc,
                "id":           ag_id,
                "name":         ag.get("name") or None,
                "owner":        owner or None,
                "score":        float(DF_SCO.loc[ag_id, "score"]) if ag_id in DF_SCO.index else 0.0,
                "shared_with":  members or [],
                "visibility":   "group",
                "model":        dom_model.get(ag_id),
                "prompt":       ag.get("prompt", ""),
                "temperature":  (None if pd.isna(ag.get("temperature")) else float(ag.get("temperature"))) if ("temperature" in ag) else None,
            }
            for k in ["score", "temperature"]:
                v = rec[k]
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
            records.append(rec)

        out_path = out_dir / f"groups_enriched_{_slug(group_id)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        if args.verbose:
            print(f"[WRITE] {out_path} → {len(records)} agents")

        with open(diagnostics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"group_id": group_id, "n_agents": len(records)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
