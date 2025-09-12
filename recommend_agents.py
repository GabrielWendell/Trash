#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA Agent Recommendation System (Popularity + Recency)
======================================================

Goal
----
Given (i) EVA access logs and (ii) YAML files describing public agents, 
compute a simple, robust recommendation list of the most accessed agents, 
optionally modulated by recency.

Input
-----
1) Logs: files like `logs-2025-08-04-10-33-48_junho.log` (or any `logs-*.log`).
   Each line is assumed to *possibly* contain an agent identifier.
   We support multiple regex patterns to extract the agent_id from lines.
   Timestamps are inferred from the log filename and/or line content (if available).

2) YAMLs: files at path pattern
   `Agents Chatbots/s3_agents_download/public/<owner_email>/*.yaml|*.YAML`
   Each YAML must contain agent metadata. We tolerate both English and Portuguese keys:
     - English: {agent_id, agent_name, initial_msg, prompt, temp}
     - Portuguese: {id_agente, nome_agente, msg_inicial, prompt, temp}

Output
------
- Top-K recommendations overall (CSV + JSON) with columns:
  [rank, agent_id, agent_name, owner_email, access_count, last_access_iso,
   popularity, recency, score]

Scoring
-------
Let c_i be the access count for agent i, and t_i the timestamp of last access.
Let C_max = max_j c_j. Define normalized popularity p_i = c_i / C_max.
Let Δt_i = (t_now - t_i) in seconds. For a time-scale τ (in seconds), define
recency weight: w_i = exp(-Δt_i / τ). If t_i is unknown, set w_i = 0.
Combine with λ ∈ [0,1]:

    score_i = λ * p_i + (1 - λ) * w_i

This is convex and scale-invariant in counts, with exponential recency decay.

Usage
-----
python recommend_agents.py \
  --logs "/path/to/logs_dir" \
  --yaml-base "Agents Chatbots/s3_agents_download/public" \
  --topk 10 \
  --lambda 0.7 \
  --tau-days 14 \
  --since "2025-06-01" \
  --exclude-owner you@example.com \
  --out-csv recommendations_top10.csv \
  --out-json recommendations_top10.json

Notes
-----
- If a log line contains no recognizable agent_id, it is ignored.
- If YAML for an accessed agent_id is missing, the agent is still ranked but
  name/owner may be empty; will appear as "<unknown>" fields in the output.
- Supports both .yaml and .YAML.
"""

from __future__ import annotations
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # PyYAML
except Exception as e:
    print("ERROR: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    raise

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class AgentMeta:
    agent_id: str
    agent_name: str
    owner_email: str
    initial_msg: Optional[str] = None
    prompt: Optional[str] = None
    temp: Optional[float] = None

@dataclass
class AgentStats:
    count: int = 0
    last_access: Optional[datetime] = None

# -----------------------------
# Helpers
# -----------------------------

_AGENT_PATTERNS: List[re.Pattern] = [
    re.compile(r"agent_id=([A-Za-z0-9_\-:]+)"),
    re.compile(r'"agent_id"\s*:\s*"([^"]+)"'),
    re.compile(r"\bagent=([A-Za-z0-9_\-:]+)\b"),
]

_LOG_TS_IN_NAME = re.compile(
    r"logs-(?P<Y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})-(?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})"
)

def _infer_ts_from_filename(path: str) -> Optional[datetime]:
    m = _LOG_TS_IN_NAME.search(os.path.basename(path))
    if not m:
        return None
    try:
        parts = {k: int(v) for k, v in m.groupdict().items()}
        return datetime(
            parts["Y"], parts["m"], parts["d"], parts["H"], parts["M"], parts["S"], tzinfo=timezone.utc
        )
    except Exception:
        return None

def _parse_line_for_agent(line: str) -> Optional[str]:
    for pat in _AGENT_PATTERNS:
        m = pat.search(line)
        if m:
            return m.group(1).strip()
    return None

def _parse_line_for_timestamp(line: str) -> Optional[datetime]:
    """Best-effort timestamp parsing within the line.
    Accepts leading ISO-8601 timestamps (e.g., 2025-08-04T10:33:48Z) or common log prefixes.
    Returns UTC-aware datetime if possible.
    """
    # ISO-like
    iso = re.search(r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?)", line)
    if iso:
        ts = iso.group(1)
        try:
            # Normalize Z
            if ts.endswith('Z'):
                ts = ts[:-1] + '+00:00'
            # Add colon in offset if missing
            if re.search(r"[+-]\d{4}$", ts):
                ts = ts[:-2] + ':' + ts[-2:]
            dt = datetime.fromisoformat(ts)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return None

# -----------------------------
# YAML ingestion
# -----------------------------

def _coerce_agent_meta(d: dict, owner_email: str) -> Optional[AgentMeta]:
    # tolerate EN/PT keys
    agent_id = d.get('agent_id') or d.get('id_agente')
    if not agent_id:
        return None
    agent_name = d.get('agent_name') or d.get('nome_agente') or ''
    initial_msg = d.get('initial_msg') or d.get('msg_inicial')
    prompt = d.get('prompt')
    temp = d.get('temp')
    try:
        temp = float(temp) if temp is not None else None
    except Exception:
        temp = None
    return AgentMeta(
        agent_id=str(agent_id), agent_name=str(agent_name), owner_email=owner_email,
        initial_msg=initial_msg, prompt=prompt, temp=temp
    )

def load_yaml_metadata(yaml_base: str) -> Dict[str, AgentMeta]:
    """Scan owners under yaml_base and load agent metadata from YAMLs.
    Returns mapping agent_id -> AgentMeta.
    """
    patterns = [
        os.path.join(yaml_base, '*', '*.yaml'),
        os.path.join(yaml_base, '*', '*.YAML'),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    meta: Dict[str, AgentMeta] = {}
    for fp in files:
        owner = os.path.basename(os.path.dirname(fp))
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            m = _coerce_agent_meta(data, owner)
            if m:
                meta[m.agent_id] = m
        except Exception as e:
            # Skip unreadable YAML but continue
            print(f"WARN: failed to read YAML {fp}: {e}", file=sys.stderr)
            continue
    return meta

# -----------------------------
# Log parsing & aggregation
# -----------------------------

def iter_log_files(logs_dir: str, since: Optional[datetime]) -> Iterable[str]:
    for fp in sorted(glob.glob(os.path.join(logs_dir, 'logs-*.log'))):
        if since is not None:
            ts = _infer_ts_from_filename(fp)
            # If we can infer a timestamp and it's older than `since`, skip
            if ts and ts < since.astimezone(timezone.utc):
                continue
        yield fp

def aggregate_accesses(logs_dir: str, since: Optional[datetime] = None) -> Dict[str, AgentStats]:
    stats: Dict[str, AgentStats] = {}
    for fp in iter_log_files(logs_dir, since):
        file_ts = _infer_ts_from_filename(fp)
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    agent_id = _parse_line_for_agent(line)
                    if not agent_id:
                        continue
                    st = stats.setdefault(agent_id, AgentStats())
                    st.count += 1
                    # pick the most precise timestamp we can find
                    line_ts = _parse_line_for_timestamp(line) or file_ts
                    if line_ts is not None:
                        if st.last_access is None or line_ts > st.last_access:
                            st.last_access = line_ts
        except FileNotFoundError:
            print(f"WARN: log file not found: {fp}", file=sys.stderr)
        except Exception as e:
            print(f"WARN: failed to read {fp}: {e}", file=sys.stderr)
    return stats

# -----------------------------
# Scoring
# -----------------------------

def score_agents(
    stats: Dict[str, AgentStats],
    now: datetime,
    tau: timedelta,
    lam: float,
) -> Dict[str, Dict[str, float]]:
    if not stats:
        return {}
    cmax = max((s.count for s in stats.values()), default=1)
    out: Dict[str, Dict[str, float]] = {}
    tau_sec = max(tau.total_seconds(), 1.0)
    for aid, s in stats.items():
        p = (s.count / cmax) if cmax > 0 else 0.0
        if s.last_access is None:
            w = 0.0
        else:
            dt = (now - s.last_access).total_seconds()
            dt = max(dt, 0.0)
            w = math.exp(-dt / tau_sec)
        score = lam * p + (1.0 - lam) * w
        out[aid] = {"popularity": p, "recency": w, "score": score}
    return out

# -----------------------------
# Recommendation assembly
# -----------------------------

def assemble_recommendations(
    scored: Dict[str, Dict[str, float]],
    stats: Dict[str, AgentStats],
    meta: Dict[str, AgentMeta],
    topk: int,
    exclude_owner: Optional[str] = None,
) -> List[dict]:
    rows: List[Tuple[str, float]] = sorted(
        ((aid, v["score"]) for aid, v in scored.items()), key=lambda x: x[1], reverse=True
    )
    recs: List[dict] = []
    k = 0
    for aid, sc in rows:
        m = meta.get(aid)
        owner = m.owner_email if m else ''
        if exclude_owner and owner.lower() == exclude_owner.lower():
            continue
        name = m.agent_name if m and m.agent_name else '<unknown>'
        count = stats.get(aid).count if aid in stats else 0
        last_ts = stats.get(aid).last_access if aid in stats else None
        recs.append({
            "rank": len(recs) + 1,
            "agent_id": aid,
            "agent_name": name,
            "owner_email": owner or '<unknown>',
            "access_count": count,
            "last_access_iso": last_ts.astimezone(timezone.utc).isoformat() if last_ts else None,
            "popularity": scored[aid]["popularity"],
            "recency": scored[aid]["recency"],
            "score": scored[aid]["score"],
        })
        k += 1
        if k >= topk:
            break
    return recs

# -----------------------------
# I/O
# -----------------------------

def write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return
    cols = [
        'rank','agent_id','agent_name','owner_email','access_count','last_access_iso',
        'popularity','recency','score'
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

def write_json(rows: List[dict], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EVA Agent Recommender (logs + YAML)")
    p.add_argument('--logs', required=True, help='Directory containing logs-*.log files')
    p.add_argument('--yaml-base', required=True, help='Base dir of public YAMLs (owners as subdirs)')
    p.add_argument('--topk', type=int, default=10, help='Number of agents to recommend (default 10)')
    p.add_argument('--tau-days', type=float, default=14.0, help='Recency half-life scale in days (default 14)')
    p.add_argument('--lambda', dest='lam', type=float, default=0.7, help='Weight for popularity vs recency (default 0.7)')
    p.add_argument('--since', type=str, default=None, help='Only consider logs at or after this ISO date (e.g., 2025-06-01)')
    p.add_argument('--exclude-owner', type=str, default=None, help='Exclude agents owned by this email from the ranking')
    p.add_argument('--out-csv', type=str, default='recommendations_topk.csv')
    p.add_argument('--out-json', type=str, default='recommendations_topk.json')
    return p.parse_args(argv)

# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logs_dir = args.logs
    yaml_base = args.yaml_base
    topk = max(1, int(args.topk))
    lam = float(args.lam)
    lam = 0.0 if lam < 0.0 else 1.0 if lam > 1.0 else lam
    tau_days = max(0.01, float(args.tau_days))
    tau = timedelta(days=tau_days)

    since_dt = None
    if args.since:
        try:
            # Parse date-only as UTC midnight
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.since):
                since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                s = args.since
                if s.endswith('Z'):
                    s = s[:-1] + '+00:00'
                if re.search(r"[+-]\d{4}$", s):
                    s = s[:-2] + ':' + s[-2:]
                since_dt = datetime.fromisoformat(s).astimezone(timezone.utc)
        except Exception:
            print(f"WARN: Could not parse --since '{args.since}', ignoring.", file=sys.stderr)
            since_dt = None

    print("[1/4] Loading YAML metadata…", file=sys.stderr)
    meta = load_yaml_metadata(yaml_base)
    print(f"Loaded {len(meta)} agents from YAML.", file=sys.stderr)

    print("[2/4] Aggregating accesses from logs…", file=sys.stderr)
    stats = aggregate_accesses(logs_dir, since=since_dt)
    print(f"Found {len(stats)} agents in logs.", file=sys.stderr)

    print("[3/4] Scoring agents (λ={lam:.2f}, τ={tau_days} days)…", file=sys.stderr)
    now = datetime.now(timezone.utc)
    scored = score_agents(stats, now=now, tau=tau, lam=lam)

    print(f"[4/4] Assembling top-{topk} recommendations…", file=sys.stderr)
    recs = assemble_recommendations(scored, stats, meta, topk=topk, exclude_owner=args.exclude_owner)

    write_csv(recs, args.out_csv)
    write_json(recs, args.out_json)

    print(f"Done. Wrote {len(recs)} rows to:\n  - {args.out_csv}\n  - {args.out_json}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
