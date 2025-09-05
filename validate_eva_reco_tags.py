# -*- coding: utf-8 -*-
"""
Standalone validator for EVA Recommended Agents & Tagging — with **public** and **private** scopes.

This script reads YAML agents from disk (no FE, no S3), computes content-based
recommendations (nearest neighbors) and suggested tags, *separately per scope*,
and saves artifacts you can show to your tutors.

USAGE EXAMPLES (Windows / POSIX):

    # If you have the base folder that contains both 'public' and 'private'
    python validate_eva_reco_tags_scoped.py \
        --root-base "data/Agents Chatbots/s3_agents_download" \
        --out artifacts --k 8

    # Or give explicit roots for each scope
    python validate_eva_reco_tags_scoped.py \
        --public-root "data/Agents Chatbots/s3_agents_download/public" \
        --private-root "data/Agents Chatbots/s3_agents_download/private" \
        --out artifacts

OUTPUT STRUCTURE:
    artifacts/
      public/
        agents_summary_public.csv
        cluster_terms_public.json
        neighbors_public_index.json
        neighbors/ <one JSON per agent id>
        tags_public.json
      private/
        agents_summary_private.csv
        cluster_terms_private.json
        neighbors_private_index.json
        neighbors/ <one JSON per agent id>
        tags_private.json

DEPENDENCIES: pyyaml, numpy, pandas, scikit-learn.
"""
from __future__ import annotations
import os, re, json, math, argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import unicodedata
import codecs

# ------------------------- config -------------------------
TOKEN_RE = re.compile(r"[A-Za-z\d_#\.\-]+", re.UNICODE)
STOP = set("""
    a an and are as at be by for from has have he her in into is it its of on or that the to was were will with
    de da do dos das para em no na nos nas e o a os as com sem por uma um umas uns
""".split())

# Regex rules to capture strong tech signals
REGEX_RULES = [
    (re.compile(r"\bselect\b|\bfrom\b|\bjoin\b|\bwhere\b|\bgroup\b", re.IGNORECASE), ["sql"]),
    (re.compile(r"\bs3://|\bboto3\b", re.IGNORECASE), ["aws", "s3"]),
    (re.compile(r"\bathena(connection)?\b", re.IGNORECASE), ["aws", "athena"]),
    (re.compile(r"\bglue\b", re.IGNORECASE), ["aws", "glue"]),
    (re.compile(r"\bredshift\b", re.IGNORECASE), ["aws", "redshift"]),
    (re.compile(r"\bteradata\b", re.IGNORECASE), ["teradata"]),
    (re.compile(r"\bpandas\b", re.IGNORECASE), ["python", "pandas"]),
    (re.compile(r"\b(cronos|modelmonitoring|cronosproject)\b", re.IGNORECASE), ["cronos", "monitoring"]),
    (re.compile(r"\bocr\b", re.IGNORECASE), ["ocr"]),
    (re.compile(r"\bsox\b", re.IGNORECASE), ["sox"]),
    (re.compile(r"\bcompliance\b", re.IGNORECASE), ["compliance"]),
    (re.compile(r"\b(llm|rag)\b", re.IGNORECASE), ["llm", "rag"]),
]

TAXONOMY = {
    "domain": ["risk", "compliance", "sox", "aml", "credit", "monitoring", "analytics", "onboarding", "legal"],
    "data":   ["sql", "python", "r", "pandas", "aws", "s3", "athena", "glue", "redshift", "api", "ocr", "airflow"],
    "task":   ["analysis", "reporting", "document-gen", "classification", "summarization", "evaluation", "forecasting", "translation", "flowchart"],
}

# ------------------------- helpers -------------------------

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(text: str) -> str:
    t = text or ""
    # decode sequences like "\xE7" → "ç" if present
    if "\\x" in t:
        try:
            t = codecs.decode(t, "unicode_escape")
        except Exception:
            pass
    t = strip_accents(t.lower())
    toks = [w for w in TOKEN_RE.findall(t) if w not in STOP and len(w) >= 3]
    return " ".join(toks)


def crawl_yaml(root: Path, force_scope: str) -> pd.DataFrame:
    rows: List[dict] = []
    for yml in root.rglob("*.yml"):
        rows.append(load_yaml_row(yml, force_scope))
    for yml in root.rglob("*.yaml"):
        rows.append(load_yaml_row(yml, force_scope))
    df = pd.DataFrame([r for r in rows if r])
    if df.empty:
        raise SystemExit(f"No YAML files found under: {root}")
    # ensure id uniqueness; if absent, synthesize id from filename
    df["id_agente"].fillna(df["file_stem"], inplace=True)
    # drop duplicates by id taking the latest mtime
    df.sort_values("mtime", ascending=False, inplace=True)
    df = df.drop_duplicates(subset=["id_agente"])  # keep newest copy
    return df.reset_index(drop=True)


def load_yaml_row(path: Path, scope: str) -> dict | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    except Exception:
        return None
    owner = path.parent.name  # email folder
    return {
        "path": str(path),
        "file_stem": path.stem,
        "owner_email": owner,
        "scope": scope,
        "mtime": path.stat().st_mtime,
        "id_agente": data.get("id_agente") or data.get("agent_id") or None,
        "nome_agente": data.get("nome_agente") or data.get("agent_name") or "",
        "desc": data.get("desc") or "",
        "msg_inicial": data.get("msg_inicial") or data.get("initial_msg") or "",
        "prompt": data.get("prompt") or "",
        "temp": data.get("temp"),
    }


def vectorize(texts: List[str], n_components: int = 256) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.85)
    X = tfidf.fit_transform(texts)
    n_components = min(n_components, max(2, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xt = svd.fit_transform(X)
    return Xt, tfidf, svd


def kmeans_cluster(X: np.ndarray, n_docs: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(2, int(math.ceil(math.sqrt(n_docs))))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def cluster_terms_ctfidf(texts: List[str], labels: np.ndarray, top: int = 8) -> Dict[int, List[str]]:
    clusters: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(i)
    agg_docs = [" ".join(texts[i] for i in idxs) for _, idxs in sorted(clusters.items())]
    tv = TfidfVectorizer(ngram_range=(1,3), min_df=1, stop_words=STOP, token_pattern = r"[A-Za-z\d_#\.\-]+")
    M = tv.fit_transform(agg_docs)
    names = tv.get_feature_names_out()
    out: Dict[int, List[str]] = {}
    for c, row in enumerate(M.toarray()):
        order = np.argsort(-row)
        # filter out very short or stopwordy terms again
        kept = [names[j] for j in order if len(names[j]) >= 3 and names[j] not in STOP]
        out[c] = kept[:top]
    return out


def local_top_terms(tfidf: TfidfVectorizer, sv_text: str, top: int = 10) -> List[str]:
    response = tfidf.transform([sv_text])
    arr = response.toarray()[0]
    if arr.size == 0:
        return []
    idx = np.argsort(-arr)[:top]
    return [tfidf.get_feature_names_out()[i] for i in idx]


def neighbors_cosine(X: np.ndarray, k: int = 8) -> List[List[int]]:
    sims = cosine_similarity(X)
    n = X.shape[0]
    nbrs: List[List[int]] = []
    for i in range(n):
        order = np.argsort(-sims[i])  # descending
        order = [j for j in order if j != i]
        nbrs.append(order[:k])
    return nbrs


def suggest_tags(text_raw: str, cluster_terms: List[str], local_terms: List[str], top: int = 5) -> List[str]:
    cand = set()
    cand.update(cluster_terms[:5])
    cand.update(local_terms[:10])

    # keep only alnum/underscore/dot/hyphen; drop stopwords/short tokens
    filt = []
    for t in cand:
        tt = t.strip().lower().replace(" ", "-")
        if len(tt) < 3 or tt in STOP:
            continue
        if not re.fullmatch(r"[a-z0-9_\.\-]+", tt):
            continue
        filt.append(tt)
    cand = set(filt)

    low = text_raw.lower()
    for rx, tg in REGEX_RULES:
        if rx.search(low):
            cand.update(tg)
    for group in TAXONOMY.values():
        for t in group:
            if t in low:
                cand.add(t)
    out = [t.strip().lower().replace(" ", "-") for t in cand if t]
    out = [t for t in out if len(t) >= 2]
    # rank by simple heuristic: regex hits > cluster > local > taxonomy
    rank = {t: 0 for t in out}
    for t in out:
        if any(t in tg for _, tg in REGEX_RULES if _):
            rank[t] += 3
        if t in cluster_terms:
            rank[t] += 2
        if t in local_terms:
            rank[t] += 1
        if any(t in group for group in TAXONOMY.values()):
            rank[t] += 0.5
    out_sorted = sorted(out, key=lambda z: (-rank[z], z))
    return out_sorted[:top]

# ------------------------- pipeline per scope -------------------------

def process_scope(scope_name: str, root: Path, out_dir: Path, k: int, components: int) -> dict:
    print(f"\n=== [{scope_name.upper()}] Processing root: {root} ===")
    df = crawl_yaml(root, force_scope=scope_name)
    print(f"  - Agents found: {len(df)} (unique by id)")

    # Build corpus and vectors
    df["text_norm"] = [norm(f"{r['nome_agente']}\n{r['desc']}\n{r['msg_inicial']}\n{r['prompt']}") for _, r in df.iterrows()]
    X, tfidf, svd = vectorize(df["text_norm"].tolist(), n_components=components)

    # Clustering and labels
    labels, centers = kmeans_cluster(X, len(df))
    df["cluster"] = labels
    cterms = cluster_terms_ctfidf(df["text_norm"].tolist(), labels, top=8)

    # Neighbors
    nbrs = neighbors_cosine(X, k=k)

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"cluster_terms_{scope_name}.json").write_text(
        json.dumps({int(k): v for k, v in cterms.items()}, ensure_ascii=False, indent=2), encoding="utf8")

    neighbors_dir = out_dir / "neighbors"
    neighbors_dir.mkdir(exist_ok=True)
    id_list = df["id_agente"].tolist()
    neighbors_index = {}
    for i, aid in enumerate(id_list):
        nn_ids = [id_list[j] for j in nbrs[i]]
        neighbors_index[aid] = nn_ids
        (neighbors_dir / f"{aid}.json").write_text(json.dumps(nn_ids, ensure_ascii=False, indent=2), encoding="utf8")
    (out_dir / f"neighbors_{scope_name}_index.json").write_text(
        json.dumps(neighbors_index, ensure_ascii=False, indent=2), encoding="utf8")

    # Tags
    tags_map: Dict[str, List[str]] = {}
    for i, row in df.iterrows():
        c = int(row["cluster"])
        local_terms = local_top_terms(tfidf, row["text_norm"], top=12)
        text_raw = f"{row['nome_agente']}\n{row['desc']}\n{row['msg_inicial']}\n{row['prompt']}"
        tags = suggest_tags(text_raw, cterms.get(c, []), local_terms, top=5)
        tags_map[row["id_agente"]] = tags
    (out_dir / f"tags_{scope_name}.json").write_text(
        json.dumps(tags_map, ensure_ascii=False, indent=2), encoding="utf8")

    # Summary CSV
    df_out = df[["id_agente", "nome_agente", "owner_email", "scope", "path", "cluster"]].copy()
    df_out["tags"] = df_out["id_agente"].map(tags_map)
    df_out.to_csv(out_dir / f"agents_summary_{scope_name}.csv", index=False)

    # Console preview
    print(f"  • clusters: {df['cluster'].nunique()} | artifacts: {out_dir.resolve()}")
    cl_sizes = df.groupby("cluster").size().sort_values(ascending=False)
    for c, sz in cl_sizes.head(5).items():
        terms = ", ".join(cterms.get(int(c), [])[:6])
        sample = df.loc[df["cluster"]==c, "nome_agente"].head(3).tolist()
        print(f"    - Cluster {c} (n={sz}) — terms: {terms}")
        for s in sample:
            print(f"       · {s}")
    print("  • sample recs & tags:")
    for i in range(min(3, len(df))):
        aid = df.loc[i, "id_agente"]
        name = df.loc[i, "nome_agente"]
        nn = neighbors_index.get(aid, [])[:k]
        name_map = dict(zip(id_list, df["nome_agente"].tolist()))
        pretty = [f"{name_map.get(x, x)} ({x})" for x in nn[:5]]
        print(f" - {name} [{aid}] → similar: {', '.join(pretty)}")
        print(f"      tags: {', '.join(tags_map.get(aid, []))}")

    return {
        "n_agents": len(df),
        "clusters": int(df['cluster'].nunique()),
        "out_dir": str(out_dir.resolve()),
    }

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate EVA Recommendations & Tags (public & private)")
    ap.add_argument("--root-base", type=str, default=None,
                    help="Base folder that contains 'public' and/or 'private' subfolders")
    ap.add_argument("--public-root", type=str, default=None, help="Root folder for public agents")
    ap.add_argument("--private-root", type=str, default=None, help="Root folder for private agents")
    ap.add_argument("--k", type=int, default=8, help="Top-k neighbors")
    ap.add_argument("--components", type=int, default=256, help="SVD components for TF-IDF")
    ap.add_argument("--out", type=str, default="artifacts", help="Output directory for artifacts")
    args = ap.parse_args()

    # Resolve roots from arguments
    pub_root = Path(args.public_root) if args.public_root else None
    prv_root = Path(args.private_root) if args.private_root else None
    if args.root_base:
        base = Path(args.root_base)
        if (base / "public").exists():
            pub_root = pub_root or (base / "public")
        if (base / "private").exists():
            prv_root = prv_root or (base / "private")

    if not pub_root and not prv_root:
        raise SystemExit("Provide --root-base or at least one of --public-root / --private-root")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if pub_root and pub_root.exists():
        process_scope("public", pub_root, out / "public", k=args.k, components=args.components)
    else:
        print("[WARN] Public root not found or not provided — skipping.")

    if prv_root and prv_root.exists():
        process_scope("private", prv_root, out / "private", k=args.k, components=args.components)
    else:
        print("[WARN] Private root not found or not provided — skipping.")

    print(f"\nDone. Check artifacts under: {out.resolve()}\n")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nHint: pass --root-base 'data/Agents Chatbots/s3_agents_download' (quotes needed if path has spaces).\n")
    main()
