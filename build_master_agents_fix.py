#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master JSON post-processor (v2)
===============================
- Fills missing agent `name` from the single group label (if exactly one).
- Flattens `shared_with` so it becomes just a list of emails (union across all groups).

Input : <out-dir>/agents_master.json
Output: <out-dir>/agents_master_fixed.json

Usage:
    python build_master_agents_fix_v2.py --out-dir results_master --verbose
"""
import json
from pathlib import Path


def _flatten_shared_with(shared_with):
    """shared_with may be a dict {group_name: [emails]}. Return flat, deduped list.
    If it's already a list, normalize & dedupe. If None/other → empty list.
    """
    flat = []
    if isinstance(shared_with, dict):
        for members in shared_with.values():
            if isinstance(members, list):
                flat.extend(members)
    elif isinstance(shared_with, list):
        flat = list(shared_with)
    # Normalize emails: strip + lowercase; drop empties
    normed = []
    for m in flat:
        if not isinstance(m, str):
            continue
        s = m.strip().lower()
        if s:
            normed.append(s)
    # Dedup + sort
    return sorted(set(normed))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fix names and flatten shared_with in agents_master.json")
    ap.add_argument("--out-dir", required=True, help="Directory containing agents_master.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    in_path = out_dir / "agents_master.json"
    if not in_path.exists():
        raise FileNotFoundError(f"Master file not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed = []
    filled_from_group = 0
    for rec in data:
        # 1) Fill name from a single group label if missing
        if (not rec.get("name")) and isinstance(rec.get("shared_with"), dict):
            keys = list(rec["shared_with"].keys())
            if len(keys) == 1:
                rec["name"] = keys[0]
                filled_from_group += 1
        # 2) Flatten shared_with to a list of emails
        rec["shared_with"] = _flatten_shared_with(rec.get("shared_with"))
        fixed.append(rec)

    out_path = out_dir / "agents_master_fixed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    if args.verbose:
        null_names = sum(1 for r in fixed if not r.get("name"))
        print(f"[OK] Wrote {out_path} | agents={len(fixed)} | names_filled={filled_from_group} | still_null_names={null_names}")


if __name__ == "__main__":
    main()
