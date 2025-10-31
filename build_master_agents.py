#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a single **master** JSON of EVA agents by X-matching:
  - results_enriched/agents_enriched_public.json
  - results_enriched/agents_enriched_private.json
  - results_groups/groups_enriched_*.json

The master file has one row per **agent id**, deduplicated across sources, and
includes whether that agent participates in one or more groups and, if so, who
it is shared with.

Output schema per agent:
  {
    "description": str|None,
    "id": str,                   # agent_id
    "name": str|None,
    "owner": str|None,           # email (lowercased)
    "username": str|None,        # derived from owner if missing
    "shared_with": dict,         # {group_name: [member_email, ...]}
    "model": str|None,
    "visibility": str|None,      # primary visibility: public > private > group
    "vis_sources": [str],        # all visibilities encountered (diagnostic)
    "prompt": str|None,
    "temperature": float|None
  }

CLI example
-----------
python build_master_agents.py \
  --enriched-dir results_enriched \
  --groups-dir results_groups \
  --out-dir results_master \
  --verbose

Notes
-----
- When the same id appears in multiple places, precedence is:
    public > private > group (for fields: owner, username, name, prompt,
    temperature, description, model). Group entries still contribute to
    `shared_with` and to `vis_sources`.
- Group name is derived from filename: groups_enriched_<slug>.json →
  replace underscores with spaces. If you prefer exact names, include them in
  groups JSON in the future.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# -------------------------- CLI --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build master EVA agents JSON by joining enriched agents and group agents")
    p.add_argument("--enriched-dir", required=True, type=str, help="Folder with agents_enriched_public/private.json")
    p.add_argument("--groups-dir", required=True, type=str, help="Folder with groups_enriched_*.json")
    p.add_argument("--out-dir", required=True, type=str, help="Folder to write master JSON + diagnostics")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# --------------------- Helpers & Normalizers ---------------------

def norm_email(e: Optional[str]) -> Optional[str]:
    if e is None:
        return None
    e = str(e).strip().lower()
    return e or None


def email_to_username(email: Optional[str]) -> Optional[str]:
    if not email:
        return None
    local = str(email).split("@", 1)[0]
    for sep in [".", "-", "_", "+"]:
        local = local.replace(sep, " ")
    tokens = [t for t in local.split() if t]
    if not tokens:
        return None
    def norm(tok: str) -> str:
        return tok.upper() if len(tok) == 1 else tok[0].upper() + tok[1:].lower()
    return " ".join(norm(t) for t in tokens)


def prefer(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue
        return v
    return None


def load_json(path: Path) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def group_name_from_filename(path: Path) -> str:
    # groups_enriched_<slug>.json → <slug>. Replace underscores with spaces.
    name = path.stem
    if name.startswith("groups_enriched_"):
        name = name[len("groups_enriched_"):]
    return name.replace("_", " ").strip()

# ----------------------- Build Master -----------------------

def main() -> None:
    args = parse_args()
    enr_dir = Path(args.enriched_dir)
    grp_dir = Path(args.groups_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load public/private (public last so it can override private on same id)
    prv = load_json(enr_dir / "agents_enriched_private.json")
    pub = load_json(enr_dir / "agents_enriched_public.json")

    # Base map by id
    base: Dict[str, dict] = {}
    def upsert_from_agent(r: dict):
        rid = str(r.get("id", "")).strip()
        if not rid:
            return
        owner = norm_email(r.get("owner"))
        username = r.get("username") or email_to_username(owner)
        rec = {
            "description": r.get("description"),
            "id": rid,
            "name": r.get("name"),
            "owner": owner,
            "username": username,
            "shared_with": {},  # group_name -> [emails]
            "model": r.get("model"),
            "visibility": None,        # will set after collecting vis_sources
            "vis_sources": set([r.get("visibility")]) if r.get("visibility") else set(),
            "prompt": r.get("prompt"),
            "temperature": r.get("temperature"),
        }
        base[rid] = rec

    # Insert private first then public to let public override the base fields
    for r in prv:
        upsert_from_agent(r)
    for r in pub:
        if str(r.get("id", "")).strip() in base:
            # merge with precedence to public values
            b = base[str(r["id"]).strip()]
            b["description"] = prefer(r.get("description"), b.get("description"))
            b["name"] = prefer(r.get("name"), b.get("name"))
            new_owner = norm_email(r.get("owner"))
            b["owner"] = prefer(new_owner, b.get("owner"))
            b["username"] = prefer(r.get("username"), email_to_username(new_owner), b.get("username"))
            b["model"] = prefer(r.get("model"), b.get("model"))
            b["prompt"] = prefer(r.get("prompt"), b.get("prompt"))
            b["temperature"] = prefer(r.get("temperature"), b.get("temperature"))
            if r.get("visibility"):
                b["vis_sources"].add(r.get("visibility"))
        else:
            upsert_from_agent(r)

    # Load groups and merge
    group_files = sorted(grp_dir.glob("groups_enriched_*.json"))
    groups_seen = 0
    ids_only_in_groups = set()
    for gf in group_files:
        groups_seen += 1
        gname = group_name_from_filename(gf)
        rows = load_json(gf)
        for r in rows:
            rid = str(r.get("id", "")).strip()
            if not rid:
                continue
            # Members and visibility contribution
            members = r.get("shared_with") or []
            members = sorted({m.strip().lower() for m in members if isinstance(m, str) and m.strip()})
            if rid not in base:
                ids_only_in_groups.add(rid)
                owner = norm_email(r.get("owner"))
                base[rid] = {
                    "description": r.get("description"),
                    "id": rid,
                    "name": r.get("name"),
                    "owner": owner,
                    "username": email_to_username(owner),
                    "shared_with": {},
                    "model": r.get("model"),
                    "visibility": None,
                    "vis_sources": set(),
                    "prompt": r.get("prompt"),
                    "temperature": r.get("temperature"),
                }
            b = base[rid]
            # Prefer existing (public/private) fields; only fill if missing
            b["description"] = prefer(b.get("description"), r.get("description"))
            b["name"] = prefer(b.get("name"), r.get("name"))
            new_owner = norm_email(r.get("owner"))
            b["owner"] = prefer(b.get("owner"), new_owner)
            b["username"] = prefer(b.get("username"), email_to_username(new_owner))
            b["model"] = prefer(b.get("model"), r.get("model"))
            b["prompt"] = prefer(b.get("prompt"), r.get("prompt"))
            b["temperature"] = prefer(b.get("temperature"), r.get("temperature"))
            # Visibility accumulation
            if r.get("visibility"):
                b["vis_sources"].add(r.get("visibility"))
            # Group sharing
            if members:
                b["shared_with"][gname] = members

    # Finalize visibility + materialize types
    def choose_primary_visibility(srcs: Iterable[str]) -> Optional[str]:
        srcs = set([s for s in srcs if s])
        if "public" in srcs:
            return "public"
        if "private" in srcs:
            return "private"
        if "group" in srcs:
            return "group"
        return None

    for rec in base.values():
        rec["visibility"] = choose_primary_visibility(rec.get("vis_sources", []))
        # Convert set → sorted list for diagnostics
        rec["vis_sources"] = sorted(list(rec.get("vis_sources", [])))
        # Normalize owner/username once more
        rec["owner"] = norm_email(rec.get("owner"))
        if not rec.get("username") and rec.get("owner"):
            rec["username"] = email_to_username(rec["owner"])
        # NEW: name fallback from single group label
        if (not rec.get("name")) and isinstance(rec.get("shared_with"), dict) and len(rec["shared_with"]) == 1:
            try:
                rec["name"] = next(iter(rec["shared_with"].keys()))
            except Exception:
                pass

    # Emit master and diagnostics
    master = sorted(base.values(), key=lambda x: (x.get("visibility") or "zzz", x.get("name") or "", x.get("id") or ""))

    out_master = out_dir / "agents_master.json"
    with open(out_master, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)

    # Diagnostics
    vis_counts = Counter([m.get("visibility") for m in master])
    diag = {
        "n_master": len(master),
        "n_public_private_input": len(pub) + len(prv),
        "n_group_files": groups_seen,
        "n_only_in_groups": len(ids_only_in_groups),
        "visibility_breakdown": dict(vis_counts),
    }
    out_diag = out_dir / "agents_master_diagnostics.json"
    with open(out_diag, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"[WRITE] {out_master}  (rows={len(master)})")
        print(f"[WRITE] {out_diag}")


if __name__ == "__main__":
    main()
