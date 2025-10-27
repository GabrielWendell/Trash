#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed version of build_master_agents_fix.py.
--------------------------------------------
Removes the unnecessary imports from build_master_agents that caused:
    ImportError: cannot import name 'norm_email' from 'build_master_agents'

This script simply loads agents_master.json, infers missing names from
shared_with group names, and saves agents_master_fixed.json.
"""

import json
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix missing agent names in master JSON.")
    parser.add_argument("--out-dir", required=True, help="Directory containing agents_master.json")
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
