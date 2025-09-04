# -*- coding: utf-8 -*-
"""Recommender index builder (YAML-only MVP).
Produces lightweight JSON artifacts in S3 so the Streamlit app can serve
recommendations without heavy CPU.

Artifacts written under agents/index/ :
  neighbors/<id>.json            # list of similar agent ids (by cosine)
  cluster_terms.json             # {cluster_id: [top terms]}
  cluster_representatives.json   # [agent_id,...] centroid-closest per cluster

Vectorization: default TF-IDF (1–3 grams) -> TruncatedSVD(256) for speed.
Swap to sentence embeddings by replacing `vectorize()`.
"""
from __future__ import annotations
import os, json, math, re
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from s3_base import AgentManager
import yaml

TOKEN_RE = re.compile(r"[A-Za-z\d_#\.\-]+")
STOP = set("""a an and are as at be by for from has he in is it its of on or that the to was were will with de da do dos das para em no na nos nas e o a os as""".split())


def normalize(text: str) -> str:
    t = (text or "").lower()
    toks = [w for w in TOKEN_RE.findall(t) if w not in STOP]
    return " ".join(toks)


def load_agents_text(am: AgentManager) -> Tuple[List[str], List[str]]:
    ids = am.list_all_agents()
    texts: List[str] = []
    kept_ids: List[str] = []
    for i in ids:
        ag = am.get_agent(i)
        if isinstance(ag, str):
            continue
        doc = f"{ag.nome_agente}\n{ag.desc}\n{ag.msg_inicial}\n{ag.prompt}"
        texts.append(normalize(doc))
        kept_ids.append(i)
    return kept_ids, texts


def vectorize(texts: List[str]) -> np.ndarray:
    tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.8)
    X = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(256, max(32, X.shape[1]//4)))
    return svd.fit_transform(X)


def cluster(X: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
    k = max(2, int(math.ceil(math.sqrt(n_docs))))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return labels, centers


def c_tfidf(texts: List[str], labels: np.ndarray) -> Dict[int, List[str]]:
    clusters = {}
    k = labels.max() + 1
    for c in range(k):
        docs = [texts[i] for i in range(len(texts)) if labels[i]==c]
        if not docs:
            clusters[c] = []
            continue
        tv = TfidfVectorizer(ngram_range=(1,3), min_df=1)
        M = tv.fit_transform([" ".join(docs)])  # aggregate
        terms = sorted(zip(tv.get_feature_names_out(), M.toarray()[0]), key=lambda x: x[1], reverse=True)
        clusters[c] = [t for t,_ in terms[:8]]
    return clusters


def build_and_push(bucket: str, user_email: str):
    am = AgentManager(bucket=bucket, user_mail=user_email)
    ids, texts = load_agents_text(am)
    if not ids:
        return
    X = vectorize(texts)
    labels, centers = cluster(X, len(ids))

    # neighbors
    sims = cosine_similarity(X)
    for i, aid in enumerate(ids):
        order = np.argsort(-sims[i])  # descending
        nn = [ids[j] for j in order if j != i][:10]
        key = f"{am.base_folder}index/neighbors/{aid}.json"
        am.client.put_object(Bucket=am.bucket, Key=key, Body=json.dumps(nn).encode("utf8"))

    # cluster terms
    terms = c_tfidf(texts, labels)
    terms_key = f"{am.base_folder}index/cluster_terms.json"
    am.client.put_object(Bucket=am.bucket, Key=terms_key, Body=json.dumps({str(k):v for k,v in terms.items()}, ensure_ascii=False).encode("utf8"))

    # representatives (closest to center)
    reps: List[str] = []
    for c in range(labels.max()+1):
        idx = [i for i,l in enumerate(labels) if l==c]
        if not idx:
            continue
        sub = X[idx]
        center = centers[c]
        d = ((sub - center)**2).sum(axis=1)
        best = idx[int(np.argmin(d))]
        reps.append(ids[best])
    reps_key = f"{am.base_folder}index/cluster_representatives.json"
    am.client.put_object(Bucket=am.bucket, Key=reps_key, Body=json.dumps(reps).encode("utf8"))


if __name__ == "__main__":
    # Example: export AWS creds in env and run
    bucket = os.environ.get("EVA_BUCKET")
    user = os.environ.get("EVA_USER_EMAIL", "user@example.com")
    build_and_push(bucket, user)