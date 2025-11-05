#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter/mask for agents_master.json
----------------------------------

This script reads a master JSON (list of agent dicts), applies the following
transformations, and writes a cleaned JSON:

1) Translate visibility values: "public" → "Público", "private" → "Privado".
   (Other values, e.g., "group", are left unchanged.)
2) Remove the field "vis_sources" from each agent, if present.
3) Drop all agents whose "prompt" field is null/empty/missing.

Usage
-----
python filter_master_agents.py \
  --input results_master/agents_master.json \
  --output results_master/agents_master_filtered.json \
  --verbose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter/mask master agents JSON: translate visibility, drop vis_sources, remove null-prompts")
    p.add_argument("--input", required=True, type=str, help="Path to master JSON (e.g., results_master/agents_master.json)")
    p.add_argument("--output", required=True, type=str, help="Path to write filtered JSON")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_nullish_prompt(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return True
        # Treat common placeholders as nullish
        if s.lower() in {"none", "null", "nan"}:
            return True
    return False


def translate_visibility(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    lv = str(v).strip().lower()
    if lv == "public":
        return "Público"
    if lv == "private":
        return "Privado"
    # Leave other visibilities unchanged (e.g., "group")
    return v


def clean_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 3) filter by prompt
    if is_nullish_prompt(rec.get("prompt")):
        return None

    # 1) visibility translation
    rec["visibility"] = translate_visibility(rec.get("visibility"))

    # 2) remove vis_sources
    if "vis_sources" in rec:
        rec.pop("vis_sources", None)

    return rec


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    data = load_json(inp)

    # Accept either a list or a dict with top-level list under a known key
    if isinstance(data, dict):
        # Try to locate a list of records
        candidates = None
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                candidates = v
                break
        if candidates is None:
            # If dict but not holding a list, consider empty
            records: List[Dict[str, Any]] = []
        else:
            records = candidates  # type: ignore
    elif isinstance(data, list):
        records = data
    else:
        records = []

    total = len(records)
    kept: List[Dict[str, Any]] = []
    dropped_null_prompt = 0

    for rec in records:
        cleaned = clean_record(dict(rec))  # work on a shallow copy
        if cleaned is None:
            dropped_null_prompt += 1
        else:
            kept.append(cleaned)

    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"[FILTER] total={total} kept={len(kept)} dropped_null_prompt={dropped_null_prompt}")
        # Quick check of visibility distribution after translation
        from collections import Counter
        vis = Counter(str(r.get("visibility")) for r in kept)
        print("[FILTER] visibility_counts:", dict(vis))


if __name__ == "__main__":
    main()
