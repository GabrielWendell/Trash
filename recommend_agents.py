#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_agents.py — Popularity/Recency + (optional) Personalization
=====================================================================

Reads your **logs2csv.py outputs** (the per-log CSVs in a `logs_csv/` folder)
and **public agent YAMLs** and produces ranked agent recommendations.

Inputs
------
- Logs CSV schema (per your converter):
  columns = [timestamp, user, page, message, __line__, type, selected_agent]
- YAML base directory (public):
  Agents Chatbots/s3_agents_download/public/<owner_email>/*.yaml|*.YAML
  Keys tolerated in YAML: EN/PT pairs —
    agent_id | id_agente, agent_name | nome_agente, initial_msg | msg_inicial, prompt, temp.

Scoring
-------
Let c_i be access count and t_i the last access time.
    popularity_i = c_i / max_j c_j
    recency_i    = exp(-(now - t_i)/tau)
    global_i     = λ * popularity_i + (1-λ) * recency_i
If --for-user is given, add a **Jaccard item-item** personalization term:
    perscore_j(user) = Σ_{i in user_agents} J(i,j),  J(i,j)=|U_i∩U_j|/|U_i∪U_j|
    perscore is min-max normalized to [0,1] over candidates.
Final score:
    final_j = α*global_j + (1-α)*perscore_j

Outputs
-------
- <out_dir>/agents_catalog.csv           (from YAMLs)
- <out_dir>/logs_consolidated.csv        (from logs_csv/*)
- <out_dir>/recommendations_topK.csv|json

Examples
--------
python recommend_agents.py \
  --logs-csv ./logs_csv \
  --yaml-base "Agents Chatbots/s3_agents_download/public" \
  --out out \
  --topk 10 \
  --lambda 0.7 \
  --tau-days 14 \
  --for-user you@domain.com \
  --alpha 0.6 \
  --novel-only \
  --exclude-owner you@domain.com \
  --verbose
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
try:
    import yaml  # PyYAML
except Exception:
    print("ERROR: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    raise

# -----------------------------
# YAML catalog
# -----------------------------
@dataclass
class AgentMeta:
    agent_id: str
    agent_name: str
    owner_email: str
    initial_msg: Optional[str] = None
    prompt: Optional[str] = None
    temp: Optional[float] = None


def _coerce_meta(d: dict, owner_email: str) -> Optional[AgentMeta]:
    aid = d.get('agent_id') or d.get('id_agente')
    if not aid:
        return None
    name = d.get('agent_name') or d.get('nome_agente') or ''
    init = d.get('initial_msg') or d.get('msg_inicial')
    prm  = d.get('prompt')
    tmp  = d.get('temp')
    try:
        tmp = float(tmp) if tmp is not None else None
    except Exception:
        tmp = None
    return AgentMeta(agent_id=str(aid), agent_name=str(name), owner_email=owner_email,
                     initial_msg=init, prompt=prm, temp=tmp)


def load_yaml_catalog(yaml_base: str, verbose: bool=False) -> pd.DataFrame:
    patterns = [os.path.join(yaml_base, '*', '*.yaml'), os.path.join(yaml_base, '*', '*.YAML')]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    metas: List[AgentMeta] = []
    for fp in files:
        owner = os.path.basename(os.path.dirname(fp))
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                d = yaml.safe_load(f) or {}
            m = _coerce_meta(d, owner)
            if m:
                metas.append(m)
        except Exception as e:
            print(f"WARN: YAML read failed for {fp}: {e}", file=sys.stderr)
    df = pd.DataFrame([m.__dict__ for m in metas])
    if verbose:
        print(f"Cataloged {len(df)} agents from YAML")
    return df

# -----------------------------
# Logs CSV loading
# -----------------------------
_LOG_AGENT_FALLBACK = [
    re.compile(r"agent_id=([A-Za-z0-9_\-:]+)"),
    re.compile(r'"agent_id"\s*:\s*"([^"]+)"'),
    re.compile(r"\bagent=([A-Za-z0-9_\-:]+)\b"),
]

def _infer_agent_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    for rx in _LOG_AGENT_FALLBACK:
        m = rx.search(text)
        if m:
            return m.group(1).strip()
    return None


def load_logs_consolidated(logs_csv_dir: str, verbose: bool=False) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(logs_csv_dir, '*.csv')))
    if not paths:
        if verbose:
            print(f"No CSVs found in {logs_csv_dir}")
        return pd.DataFrame(columns=['timestamp','user','page','message','__line__','type','selected_agent'])

    dfs = []
    for fp in paths:
        try:
            df = pd.read_csv(fp)
            # normalize columns
            df.columns = [c.strip() for c in df.columns]
            # expected columns per user spec
            for need in ['timestamp','user','page','message','__line__','type','selected_agent']:
                if need not in df.columns:
                    df[need] = pd.NA
            dfs.append(df[['timestamp','user','page','message','__line__','type','selected_agent']])
        except Exception as e:
            print(f"WARN: failed reading {fp}: {e}", file=sys.stderr)
    if not dfs:
        return pd.DataFrame(columns=['timestamp','user','page','message','__line__','type','selected_agent'])

    logs = pd.concat(dfs, ignore_index=True)

    # Parse timestamps
    try:
        logs['timestamp'] = pd.to_datetime(logs['timestamp'], utc=True, errors='coerce')
    except Exception:
        logs['timestamp'] = pd.NaT

    # Fill selected_agent via fallbacks if missing
    mask_na = logs['selected_agent'].isna() | (logs['selected_agent'].astype(str).str.strip() == '')
    if mask_na.any():
        from_msg = logs.loc[mask_na, 'message'].apply(_infer_agent_from_text)
        from_page = logs.loc[mask_na & from_msg.isna(), 'page'].apply(_infer_agent_from_text)
        logs.loc[mask_na, 'selected_agent'] = from_msg
        mask_na2 = logs['selected_agent'].isna() | (logs['selected_agent'].astype(str).str.strip() == '')
        logs.loc[mask_na2, 'selected_agent'] = from_page

    # Final cleanup
    logs['selected_agent'] = logs['selected_agent'].astype(str).str.strip()
    logs.loc[logs['selected_agent'] == '', 'selected_agent'] = pd.NA
    logs = logs.dropna(subset=['selected_agent'])

    # Standardize user
    if 'user' in logs.columns:
        logs['user'] = logs['user'].astype(str).str.strip().str.lower()

    if verbose:
        print(f"Loaded {len(logs)} log rows from {len(paths)} CSV files")
    return logs

# -----------------------------
# Aggregations & scoring
# -----------------------------

def aggregate_by_agent(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame(columns=['agent_id','access_count','last_access'])
    grp = logs.groupby('selected_agent', dropna=True)['timestamp']
    agg = grp.agg(['count','max']).reset_index()
    agg = agg.rename(columns={'selected_agent':'agent_id','count':'access_count','max':'last_access'})
    return agg


def compute_global_score(df: pd.DataFrame, lam: float, tau_days: float) -> pd.DataFrame:
    if df.empty:
        df['popularity'] = []
        df['recency'] = []
        df['global_score'] = []
        return df
    lam = min(max(float(lam), 0.0), 1.0)
    now = pd.Timestamp.utcnow()
    cmax = max(df['access_count'].max(), 1)
    df['popularity'] = df['access_count'] / cmax
    tau_sec = max(float(tau_days), 1e-3) * 24 * 3600.0

    def recency(ts: pd.Timestamp) -> float:
        if pd.isna(ts):
            return 0.0
        dt = (now - ts).total_seconds()
        if dt < 0:
            dt = 0.0
        return math.exp(-dt / tau_sec)

    df['recency'] = df['last_access'].apply(recency)
    df['global_score'] = lam*df['popularity'] + (1.0 - lam)*df['recency']
    return df

# -----------------------------
# Personalization via Jaccard item-item
# -----------------------------

def _user_item_sets(logs: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return (user->set(agent_id), agent->set(user))."""
    ui: Dict[str, Set[str]] = {}
    iu: Dict[str, Set[str]] = {}
    for user, aid in logs[['user','selected_agent']].dropna().itertuples(index=False):
        if not isinstance(user, str) or not user:
            continue
        if not isinstance(aid, str) or not aid:
            continue
        ui.setdefault(user, set()).add(aid)
        iu.setdefault(aid, set()).add(user)
    return ui, iu


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union


def personalize_scores(logs: pd.DataFrame, target_user: str, candidate_df: pd.DataFrame,
                        alpha: float) -> pd.DataFrame:
    """Add `final_score` using α*global + (1-α)*perscore for target_user.
       `candidate_df` must contain columns: agent_id, global_score.
    """
    alpha = min(max(float(alpha), 0.0), 1.0)
    if logs.empty or not target_user:
        candidate_df['final_score'] = candidate_df['global_score']
        return candidate_df

    target_user = target_user.strip().lower()
    U2A, A2U = _user_item_sets(logs)
    if target_user not in U2A:
        # no history → fall back to global only
        candidate_df['final_score'] = candidate_df['global_score']
        return candidate_df

    user_agents = U2A[target_user]

    # Precompute jaccard to user's items
    perscore: Dict[str, float] = {}
    # Candidate pool: all agents seen in logs
    all_agents = set(A2U.keys())

    # Compute Jaccard between each candidate and user's set
    for cand in all_agents:
        if cand in user_agents:
            continue
        s = 0.0
        for i in user_agents:
            s += _jaccard(A2U.get(i, set()), A2U.get(cand, set()))
        perscore[cand] = s

    if not perscore:
        candidate_df['final_score'] = candidate_df['global_score']
        return candidate_df

    # Normalize perscore to [0,1]
    vmax = max(perscore.values()) if perscore else 1.0
    if vmax <= 0:
        vmax = 1.0
    for k in list(perscore.keys()):
        perscore[k] = perscore[k] / vmax

    candidate_df = candidate_df.copy()
    candidate_df['perscore'] = candidate_df['agent_id'].map(perscore).fillna(0.0)
    candidate_df['final_score'] = alpha*candidate_df['global_score'] + (1.0 - alpha)*candidate_df['perscore']
    return candidate_df

# -----------------------------
# Main routine
# -----------------------------

def run(
    logs_csv_dir: str,
    yaml_base: str,
    out_dir: str,
    topk: int,
    lam: float,
    tau_days: float,
    for_user: Optional[str],
    alpha: float,
    exclude_owner: Optional[str],
    novel_only: bool,
    verbose: bool,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    # Load
    if verbose:
        print("[1/4] Reading logs CSVs …")
    logs = load_logs_consolidated(logs_csv_dir, verbose=verbose)

    if verbose:
        print("[2/4] Loading YAML catalog …")
    catalog = load_yaml_catalog(yaml_base, verbose=verbose)

    # Aggregate
    if verbose:
        print("[3/4] Aggregating + scoring (global) …")
    agg = aggregate_by_agent(logs)
    scored = compute_global_score(agg, lam=lam, tau_days=tau_days)

    # XMatch by agent_id → add owner & name
    merged = scored.merge(catalog, how='left', left_on='agent_id', right_on='agent_id')

    # optional: exclude owner
    if exclude_owner:
        merged = merged[~merged['owner_email'].str.lower().eq(str(exclude_owner).lower())]

    # Personalized rerank
    merged = merged[['agent_id','agent_name','owner_email','access_count','last_access','popularity','recency','global_score']]

    if for_user:
        if verbose:
            print(f"[4/4] Personalizing for {for_user} (α={alpha:.2f}) …")
        merged = personalize_scores(logs, for_user, merged, alpha=alpha)
        score_col = 'final_score'
    else:
        merged['final_score'] = merged['global_score']
        score_col = 'final_score'

    # novel-only: hide items the user already used
    if for_user and novel_only:
        used = set(logs.loc[logs['user'].eq(for_user.strip().lower()), 'selected_agent'].dropna().astype(str))
        if used:
            merged = merged[~merged['agent_id'].isin(used)]

    merged = merged.sort_values(score_col, ascending=False)

    out_cols = ['agent_id','agent_name','owner_email','access_count','last_access','popularity','recency','global_score','final_score']
    out_df = merged[out_cols].copy()
    out_df.insert(0, 'rank', range(1, len(out_df)+1))

    # Write artifacts
    catalog_path = os.path.join(out_dir, 'agents_catalog.csv')
    catalog.to_csv(catalog_path, index=False)

    logs_path = os.path.join(out_dir, 'logs_consolidated.csv')
    # Save the consolidated (even though we read per-file CSVs)
    logs.to_csv(logs_path, index=False)

    recs_csv = os.path.join(out_dir, f'recommendations_top{topk}.csv')
    out_df.head(int(topk)).to_csv(recs_csv, index=False)

    recs_json = os.path.join(out_dir, f'recommendations_top{topk}.json')
    with open(recs_json, 'w', encoding='utf-8') as f:
        json.dump(out_df.head(int(topk)).to_dict(orient='records'), f, ensure_ascii=False, indent=2, default=str)

    if verbose:
        print(f"Wrote:\n  - {catalog_path}\n  - {logs_path}\n  - {recs_csv}\n  - {recs_json}")

    return recs_csv, recs_json

# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='EVA — Recommend agents from logs_csv + YAML catalog')
    p.add_argument('--logs-csv', required=True, help='Directory with logs2csv.py outputs (CSV files)')
    p.add_argument('--yaml-base', required=True, help='Public YAML base dir')
    p.add_argument('--out', default='out', help='Output directory')
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--lambda', dest='lam', type=float, default=0.7, help='Weight for popularity vs recency')
    p.add_argument('--tau-days', type=float, default=14.0, help='Recency decay timescale (days)')
    p.add_argument('--for-user', type=str, default=None, help='(Optional) personalize for this user email')
    p.add_argument('--alpha', type=float, default=0.6, help='Blend global vs personalization (α)')
    p.add_argument('--novel-only', action='store_true', help='When personalizing, hide items already used by the user')
    p.add_argument('--exclude-owner', type=str, default=None, help='Exclude agents owned by this email')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run(
        logs_csv_dir=args.logs_csv,
        yaml_base=args.yaml_base,
        out_dir=args.out,
        topk=args.topk,
        lam=args.lam,
        tau_days=args.tau_days,
        for_user=args.for_user,
        alpha=args.alpha,
        exclude_owner=args.exclude_owner,
        novel_only=args.novel_only,
        verbose=args.verbose,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
