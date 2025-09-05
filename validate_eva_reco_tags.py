# -*- coding: utf-8 -*-
"""
Standalone validator for EVA Recommended Agents + Tagging (YAML-only, no FE).

Usage (Windows / POSIX):
    python validate_eva_reco_tags.py \
        --root "data/Agents Chatbots/s3_agents_download/public" \
        --k 8 --out artifacts

This script will:
  1) Crawl the YAMLs under --root (recursively) and build a corpus.
  2) Vectorize with TF-IDF(1–3-grams) -> TruncatedSVD(256) for speed.
  3) Cluster with K-Means (k≈ceil(sqrt(n))).
  4) Compute cluster labels via c-TF-IDF.
  5) Compute cosine nearest neighbors for each agent (top-k).
  6) Suggest tags per agent (cluster terms + local TFIDF n-grams + regex + taxonomy).
  7) Save artifacts into --out/ (CSV + JSONs) and print a readable summary.

No network access; depends on: pyyaml, numpy, pandas, scikit-learn.
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

# ------------------------- config -------------------------
TOKEN_RE = re.compile(r"[A-Za-z\d_#\.\-]+", re.UNICODE)
STOP = set("""
    a an and are as at be by for from has have he her in into is it its of on or that the to was were will with
    de da do dos das para em no na nos nas e o a os as com sem por uma um umas uns
""".split())

# Regex rules to capture strong tech signals
REGEX_RULES = [
    (re.compile(r"\bselect\b|\bfrom\b|\bjoin\b|\bwhere\b|\bgroup\b"), ["sql"]),
    (re.compile(r"\bs3://|\bboto3\b"), ["aws", "s3"]),
    (re.compile(r"\bathena(connection)?\b"), ["aws", "athena"]),
    (re.compile(r"\bglue\b"), ["aws", "glue"]),
    (re.compile(r"\bredshift\b"), ["aws", "redshift"]),
    (re.compile(r"\bteradata\b"), ["teradata"]),
    (re.compile(r"\bpandas\b"), ["python", "pandas"]),
    (re.compile(r"\b(cronos|modelmonitoring|cronosproject)\b"), ["cronos", "monitoring"]),
    (re.compile(r"\bocr\b"), ["ocr"]),
    (re.compile(r"\bsox\b"), ["sox"]),
    (re.compile(r"\bcompliance\b"), ["compliance"]),
    (re.compile(r"\b(llm|rag)\b"), ["llm", "rag"]),
]

TAXONOMY = {
    "domain": ["risk", "compliance", "sox", "aml", "credit", "monitoring", "analytics", "onboarding", "legal"],
    "data":   ["sql", "python", "r", "pandas", "aws", "s3", "athena", "glue", "redshift", "api", "ocr", "airflow"],
    "task":   ["analysis", "reporting", "document-gen", "classification", "summarization", "evaluation", "forecasting", "translation", "flowchart"],
}

# ------------------------- helpers -------------------------

def norm(text: str) -> str:
    t = (text or "").lower()
    toks = [w for w in TOKEN_RE.findall(t) if w not in STOP]
    return " ".join(toks)


def crawl_yaml(root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for yml in root.rglob("*.yml"):
        rows.append(load_yaml_row(yml))
    for yml in root.rglob("*.yaml"):
        rows.append(load_yaml_row(yml))
    df = pd.DataFrame([r for r in rows if r])
    if df.empty:
        raise SystemExit(f"No YAML files found under: {root}")
    # ensure id uniqueness; if absent, synthesize id from filename
    df["id_agente"].fillna(df["file_stem"], inplace=True)
    # drop duplicates by id taking the latest mtime
    df.sort_values("mtime", ascending=False, inplace=True)
    df = df.drop_duplicates(subset=["id_agente"])  # keep newest copy
    return df.reset_index(drop=True)


def load_yaml_row(path: Path) -> dict | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    except Exception:
        return None
    owner = path.parent.name  # email folder
    scope = path.parents[1].name  # 'public' or 'private'
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
    # aggregate documents per cluster then TF-IDF over these aggregated docs
    clusters: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(i)
    agg_docs = [" ".join(texts[i] for i in idxs) for c, idxs in sorted(clusters.items())]
    tv = TfidfVectorizer(ngram_range=(1,3), min_df=1)
    M = tv.fit_transform(agg_docs)
    names = tv.get_feature_names_out()
    out: Dict[int, List[str]] = {}
    for c, row in enumerate(M.toarray()):
        order = np.argsort(-row)[:top]
        out[c] = [names[j] for j in order]
    return out


def local_top_terms(tfidf: TfidfVectorizer, sv_text: str, top: int = 10) -> List[str]:
    # compute local TF-IDF features directly from the fitted vectorizer
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

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate EVA Recommendations & Tags (YAML-only)")
    ap.add_argument("--root", type=str, required=True,
                    help="Root folder of YAMLs (e.g., data/Agents Chatbots/s3_agents_download/public)")
    ap.add_argument("--k", type=int, default=8, help="Top-k neighbors")
    ap.add_argument("--components", type=int, default=256, help="SVD components for TF-IDF")
    ap.add_argument("--out", type=str, default="artifacts", help="Output directory for artifacts")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/7] Crawling YAMLs under: {root}")
    df = crawl_yaml(root)
    print(f"  - Agents found: {len(df)} (unique by id)")

    print("[2/7] Building corpus and vectorizing...")
    df["text_norm"] = [norm(f"{r['nome_agente']}\n{r['desc']}\n{r['msg_inicial']}\n{r['prompt']}") for _, r in df.iterrows()]
    X, tfidf, svd = vectorize(df["text_norm"].tolist(), n_components=args.components)

    print("[3/7] Clustering with K-Means...")
    labels, centers = kmeans_cluster(X, len(df))
    df["cluster"] = labels

    print("[4/7] Computing c-TFIDF cluster terms...")
    cterms = cluster_terms_ctfidf(df["text_norm"].tolist(), labels, top=8)
    (out / "cluster_terms.json").write_text(json.dumps({int(k): v for k, v in cterms.items()}, ensure_ascii=False, indent=2), encoding="utf8")

    print("[5/7] Computing cosine nearest neighbors...")
    nbrs = neighbors_cosine(X, k=args.k)

    # save neighbor lists and a compact JSON index
    neighbors_dir = out / "neighbors"
    neighbors_dir.mkdir(exist_ok=True)
    id_list = df["id_agente"].tolist()
    neighbors_index = {}
    for i, aid in enumerate(id_list):
        nn_ids = [id_list[j] for j in nbrs[i]]
        neighbors_index[aid] = nn_ids
        (neighbors_dir / f"{aid}.json").write_text(json.dumps(nn_ids, ensure_ascii=False, indent=2), encoding="utf8")
    (out / "neighbors_index.json").write_text(json.dumps(neighbors_index, ensure_ascii=False, indent=2), encoding="utf8")

    print("[6/7] Tagging (ensemble of cluster terms + local TF-IDF + regex + taxonomy)...")
    tags_map: Dict[str, List[str]] = {}
    for i, row in df.iterrows():
        c = int(row["cluster"])
        local_terms = local_top_terms(tfidf, row["text_norm"], top=12)
        text_raw = f"{row['nome_agente']}\n{row['desc']}\n{row['msg_inicial']}\n{row['prompt']}"
        tags = suggest_tags(text_raw, cterms.get(c, []), local_terms, top=5)
        tags_map[row["id_agente"]] = tags
    (out / "tags.json").write_text(json.dumps(tags_map, ensure_ascii=False, indent=2), encoding="utf8")

    print("[7/7] Exporting summary CSV...")
    df_out = df[["id_agente", "nome_agente", "owner_email", "scope", "path", "cluster"]].copy()
    df_out["tags"] = df_out["id_agente"].map(tags_map)
    df_out.to_csv(out / "agents_summary.csv", index=False)

    # -------- Pretty print quick validation --------
    print("\n=== Quick Validation Output ===")
    # Show top 5 clusters with their label terms and sample agents
    cl_sizes = df.groupby("cluster").size().sort_values(ascending=False)
    for c, sz in cl_sizes.head(5).items():
        terms = ", ".join(cterms.get(int(c), [])[:6])
        sample = df.loc[df["cluster"]==c, "nome_agente"].head(3).tolist()
        print(f"Cluster {c} (n={sz}) — terms: {terms}")
        for s in sample:
            print(f"   · {s}")

    # Show neighbors and tags for first 5 agents
    print("\nSample recommendations & tags (first 5 agents):")
    for i in range(min(5, len(df))):
        aid = df.loc[i, "id_agente"]
        name = df.loc[i, "nome_agente"]
        nn = neighbors_index.get(aid, [])[:args.k]
        print(f"- {name} [{aid}] → similar: {', '.join(nn[:5])}")
        print(f"  tags: {', '.join(tags_map.get(aid, []))}")

    print(f"\nArtifacts written to: {out.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nHint: pass --root 'data/Agents Chatbots/s3_agents_download/public' (quotes needed if path has spaces).\n")
    main()
