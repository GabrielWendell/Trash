#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_agents.py — robust XMatch v2
======================================

Fixes:
- Many `selected_agent` values in logs are **display names**, not canonical `agent_id`.
  Joining logs on YAML `agent_id` therefore produced NaNs.
- Some rows contain literal strings like "None"/"null" (UI default) → were counted as an agent.

This version:
1) Builds a lookup from YAML using **both** id and name (EN/PT), with aggressive normalization
   (lowercase, trim, remove accents, drop punctuation, collapse whitespace).
2) Maps each log row's `selected_agent` to a **canonical agent_id** via that lookup.
3) Drops rows whose `selected_agent` is empty/None/null/undefined.
4) Emits diagnostics with match rates and an `unmatched_selected_agents.csv` artifact to
   help you quickly patch missing aliases.

CLI (same flags as before):
python recommend_agents_v2.py \
  --logs-csv ./logs_csv \
  --yaml-base "Agents Chatbots/s3_agents_download/public" \
  --out out \
  --topk 10 \
  --lambda 0.7 \
  --tau-days 14 \
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
import string
import unicodedata
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
# Normalization utilities
# -----------------------------

_NULL_TOKENS = {"", "none", "null", "nan", "undefined", "<none>", "na"}
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def norm_key(s: object) -> Optional[str]:
    """Normalize an id/name for robust matching.
    - lower, strip
    - remove accents (NFKD)
    - replace punctuation with space, collapse spaces
    - turn None/"none"/"null"/etc into None
    """
    if not isinstance(s, str):
        return None
    s0 = s
    s = s.strip().lower()
    if s in _NULL_TOKENS:
        return None
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    if s in _NULL_TOKENS:
        return None
    return s

# -----------------------------
# YAML catalog → DataFrame + key map
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


def load_yaml_catalog_and_map(yaml_base: str, verbose: bool=False) -> tuple[pd.DataFrame, Dict[str, str]]:
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

    # build mapping dict from normalized id and name → canonical agent_id
    key2id: Dict[str, str] = {}
    for _, r in df.iterrows():
        k1 = norm_key(r.get('agent_id'))
        k2 = norm_key(r.get('agent_name'))
        if k1:
            key2id[k1] = r['agent_id']
        if k2:
            key2id[k2] = r['agent_id']
    if verbose:
        print(f"Cataloged {len(df)} agents from YAML; key-map size={len(key2id)}")
    return df, key2id

# -----------------------------
# Logs CSV loading
# -----------------------------
_EXPECTED = ['timestamp','user','page','message','__line__','type','selected_agent']


def load_logs(logs_csv_dir: str, verbose: bool=False) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(logs_csv_dir, '*.csv')))
    if not paths:
        if verbose:
            print(f"No CSVs found in {logs_csv_dir}")
        return pd.DataFrame(columns=_EXPECTED)

    dfs = []
    for fp in paths:
        try:
            df = pd.read_csv(fp)
            df.columns = [c.strip() for c in df.columns]
            for c in _EXPECTED:
                if c not in df.columns:
                    df[c] = pd.NA
            dfs.append(df[_EXPECTED])
        except Exception as e:
            print(f"WARN: failed reading {fp}: {e}", file=sys.stderr)
    if not dfs:
        return pd.DataFrame(columns=_EXPECTED)

    logs = pd.concat(dfs, ignore_index=True)
    # timestamp to UTC
    try:
        logs['timestamp'] = pd.to_datetime(logs['timestamp'], utc=True, errors='coerce')
    except Exception:
        logs['timestamp'] = pd.NaT
    # standardize user
    logs['user'] = logs['user'].astype(str).str.strip().str.lower()
    return logs

# -----------------------------
# Matching & aggregation
# -----------------------------

def attach_canonical_ids(logs: pd.DataFrame, key2id: Dict[str, str], out_dir: str, verbose: bool=False) -> pd.DataFrame:
    # Normalize selected_agent and map
    logs = logs.copy()
    logs['selected_agent_key'] = logs['selected_agent'].apply(norm_key)

    # mark null-like tokens
    bad = logs['selected_agent_key'].isna()
    before = len(logs)
    logs = logs[~bad]  # drop null-like
    if verbose:
        print(f"Filtered {bad.sum()} rows with null/None/undefined selected_agent; kept {len(logs)}/{before}")

    logs['agent_id'] = logs['selected_agent_key'].map(key2id)

    # diagnostics for unmatched
    unmatched = logs[logs['agent_id'].isna()]
    if not unmatched.empty:
        diag = (unmatched
                .groupby('selected_agent', dropna=True)
                .size()
                .reset_index(name='count')
                .sort_values('count', ascending=False))
        diag_path = os.path.join(out_dir, 'unmatched_selected_agents.csv')
        os.makedirs(out_dir, exist_ok=True)
        diag.to_csv(diag_path, index=False)
        if verbose:
            unique = diag['selected_agent'].nunique()
            print(f"Unmatched selected_agent values: {unique} (see {diag_path})")

    # keep only matched rows for ranking
    matched = logs.dropna(subset=['agent_id']).copy()
    return matched


def aggregate_by_agent(logs_with_ids: pd.DataFrame) -> pd.DataFrame:
    if logs_with_ids.empty:
        return pd.DataFrame(columns=['agent_id','access_count','last_access'])
    grp = logs_with_ids.groupby('agent_id', dropna=True)['timestamp']
    agg = grp.agg(['count','max']).reset_index()
    agg = agg.rename(columns={'count':'access_count','max':'last_access'})
    return agg


def score_global(agg: pd.DataFrame, lam: float, tau_days: float) -> pd.DataFrame:
    if agg.empty:
        agg['popularity'] = []
        agg['recency'] = []
        agg['global_score'] = []
        return agg
    lam = min(max(float(lam), 0.0), 1.0)
    now = pd.Timestamp.utcnow()
    cmax = max(agg['access_count'].max(), 1)
    agg['popularity'] = agg['access_count'] / cmax
    tau_sec = max(float(tau_days), 1e-3) * 86400.0

    def recency(ts: pd.Timestamp) -> float:
        if pd.isna(ts):
            return 0.0
        dt = (now - ts).total_seconds()
        if dt < 0:
            dt = 0.0
        return math.exp(-dt / tau_sec)

    agg['recency'] = agg['last_access'].apply(recency)
    agg['global_score'] = lam*agg['popularity'] + (1.0 - lam)*agg['recency']
    return agg

# -----------------------------
# Main
# -----------------------------

def run(logs_csv: str, yaml_base: str, out_dir: str, topk: int, lam: float, tau_days: float, verbose: bool) -> tuple[str,str]:
    os.makedirs(out_dir, exist_ok=True)

    # load assets
    if verbose:
        print("[1/4] Load logs CSVs …")
    logs = load_logs(logs_csv, verbose=verbose)

    if verbose:
        print("[2/4] Load YAML catalog + key map …")
    catalog, key2id = load_yaml_catalog_and_map(yaml_base, verbose=verbose)

    # map selected_agent → canonical id
    if verbose:
        print("[3/4] Map selected_agent → canonical agent_id …")
    logs_m = attach_canonical_ids(logs, key2id, out_dir, verbose=verbose)

    # rank
    agg = aggregate_by_agent(logs_m)
    scored = score_global(agg, lam=lam, tau_days=tau_days)

    # join for names/owners
    out = scored.merge(catalog, how='left', on='agent_id')
    out = out[['agent_id','agent_name','owner_email','access_count','last_access','popularity','recency','global_score']]
    out = out.sort_values('global_score', ascending=False)
    out.insert(0, 'rank', range(1, len(out)+1))

    # write
    cat_path = os.path.join(out_dir, 'agents_catalog.csv')
    catalog.to_csv(cat_path, index=False)
    logs_path = os.path.join(out_dir, 'logs_consolidated.csv')
    logs_m.to_csv(logs_path, index=False)
    recs_csv = os.path.join(out_dir, f'recommendations_top{topk}.csv')
    out.head(int(topk)).to_csv(recs_csv, index=False)
    recs_json = os.path.join(out_dir, f'recommendations_top{topk}.json')
    with open(recs_json, 'w', encoding='utf-8') as f:
        json.dump(out.head(int(topk)).to_dict(orient='records'), f, ensure_ascii=False, indent=2, default=str)

    if verbose:
        print(f"Wrote:\n  - {cat_path}\n  - {logs_path}\n  - {recs_csv}\n  - {recs_json}")
    return recs_csv, recs_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='EVA — robust XMatch recommender (logs CSV + YAML)')
    p.add_argument('--logs-csv', required=True)
    p.add_argument('--yaml-base', required=True)
    p.add_argument('--out', default='out')
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--lambda', dest='lam', type=float, default=0.7)
    p.add_argument('--tau-days', type=float, default=14.0)
    p.add_argument('--verbose', action='store_true')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args.logs_csv, args.yaml_base, args.out, args.topk, args.lam, args.tau_days, args.verbose)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
