# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json
from typing import List, Dict

from s3_base import AgentManager

# Seed taxonomy (can be moved to YAML)
TAXONOMY = {
    "domain": ["risk", "compliance", "sox", "aml", "credit", "monitoring", "analytics", "onboarding", "legal"],
    "data":   ["sql", "python", "r", "pandas", "aws", "s3", "athena", "glue", "redshift", "api", "ocr", "airflow"],
    "task":   ["analysis", "reporting", "document-gen", "classification", "summarization", "evaluation", "forecasting", "translation", "flowchart"],
}

REGEX_RULES = [
    (re.compile(r"\bselect\b|\bfrom\b|\bjoin\b"), ["sql"]),
    (re.compile(r"\bs3://|\bboto3\b"), ["aws", "s3"]),
    (re.compile(r"\bathena(connection)?\b"), ["aws", "athena"]),
    (re.compile(r"\bteradata\b"), ["teradata"]),
    (re.compile(r"\bpandas\b"), ["python", "pandas"]),
    (re.compile(r"\bcronos(project|monitoring)?\b"), ["cronos", "monitoring"]),
    (re.compile(r"\bocr\b"), ["ocr"]),
    (re.compile(r"\bsox\b"), ["sox"]),
    (re.compile(r"\bcompliance\b"), ["compliance"]),
    (re.compile(r"\b(llm|rag)\b"), ["llm", "rag"]),
]


def suggest_tags_for_text(text: str, cluster_terms: List[str]) -> List[str]:
    cand = set()
    # (1) cluster terms
    cand.update(cluster_terms[:5])
    # (2) regex
    low = text.lower()
    for rx, tg in REGEX_RULES:
        if rx.search(low):
            cand.update(t for t in tg)
    # (3) taxonomy zero-shot (cheap proxy): substring hits
    for group in TAXONOMY.values():
        for t in group:
            if t in low:
                cand.add(t)
    # keep up to 5 tags
    out = [t.strip().lower().replace(" ", "-") for t in cand if t]
    # small whitelist to avoid very generic 1-letter tokens
    out = [t for t in out if len(t) >= 2]
    return sorted(list(dict.fromkeys(out)))[:5]


def run(bucket: str, user_email: str):
    am = AgentManager(bucket=bucket, user_mail=user_email)
    cluster_terms = am.get_cluster_terms()  # {"0": [..], ...}
    # fallback
    if not cluster_terms:
        cluster_terms = {}
    all_ids = am.list_all_agents()
    for aid in all_ids:
        ag = am.get_agent(aid)
        if isinstance(ag, str):
            continue
        key = next((k for k,v in cluster_terms.items() if isinstance(v, list)), None)
        terms = []
        # naive: try to infer cluster by nearest representative is not available here; just use empty list
        if key is not None:
            terms = cluster_terms.get(key, [])
        text = f"{ag.nome_agente}\n{ag.desc}\n{ag.msg_inicial}\n{ag.prompt}"
        tags = suggest_tags_for_text(text, terms)
        if tags:
            am.add_tags(aid, tags)


if __name__ == "__main__":
    bucket = os.environ.get("EVA_BUCKET")
    user = os.environ.get("EVA_USER_EMAIL", "user@example.com")
    run(bucket, user)