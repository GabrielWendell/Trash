#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix for missing agent names in master JSON.
-----------------------------------------
This version ensures that if the 'name' field is null after merging, the script
tries to recover the agent name from the group-level data. Specifically:
- If an agent appears in exactly one group, its name is set to that group name.
- If multiple groups contain the same agent, we leave 'name' as-is (to avoid ambiguity).

Usage:
    python build_master_agents_fixed.py \
        --enriched-dir results_enriched \
        --groups-dir results_groups \
        --out-dir results_master \
        --verbose
"""

import json
from pathlib import Path
from collections import Counter
from build_master_agents import main as original_main, norm_email, email_to_username

def main():
    # We reuse the full logic of build_master_agents, then patch names after loading.
    import argparse
    parser = argparse.ArgumentParser(description="Fix missing agent names in master JSON.")
    parser.add_argument("--enriched-dir", required=True)
    parser.add_argument("--groups-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_master = Path(args.out_dir) / "agents_master.json"
    if not out_master.exists():
        raise FileNotFoundError(f"Master file not found: {out_master}")

    # Load master JSON
    with open(out_master, "r", encoding="utf-8") as f:
        master = json.load(f)

    # Fix missing names
    fixed = []
    for rec in master:
        if (not rec.get("name")) and isinstance(rec.get("shared_with"), dict):
            group_names = list(rec["shared_with"].keys())
            if len(group_names) == 1:
                rec["name"] = group_names[0]
        fixed.append(rec)

    # Save fixed master
    fixed_master = Path(args.out_dir) / "agents_master_fixed.json"
    with open(fixed_master, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    if args.verbose:
        null_names = sum(1 for r in fixed if not r.get("name"))
        print(f"[INFO] Saved {fixed_master} (agents={len(fixed)}, still null names={null_names})")

if __name__ == "__main__":
    main()
