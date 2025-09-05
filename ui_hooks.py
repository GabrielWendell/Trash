# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import List, Dict
import streamlit as st

from base.s3_base import AgentManager

# -------------------- cached getters --------------------
@st.cache_data(ttl=300)
def _am(bucket: str, user_email: str) -> AgentManager:
    return AgentManager(bucket=bucket, user_mail=user_email)

@st.cache_data(ttl=300)
def get_neighbors(bucket: str, user_email: str, agent_id: str, k: int = 8) -> List[str]:
    am = _am(bucket, user_email)
    return am.get_similar_agents(agent_id, k=k)

@st.cache_data(ttl=300)
def get_cluster_representatives(bucket: str, user_email: str, k: int = 12) -> List[str]:
    am = _am(bucket, user_email)
    return am.get_cluster_representatives(top=k)

@st.cache_data(ttl=300)
def get_tag_catalog(bucket: str, user_email: str) -> Dict[str, int]:
    am = _am(bucket, user_email)
    try:
        return am.list_tag_catalog()
    except Exception:
        return {}

# -------------------- renderers --------------------
def render_recommended_section(bucket: str, user_email: str, current_agent_id: str | None = None):
    am = _am(bucket, user_email)
    st.subheader("Recommended Agents")

    # Recommendation mode selector
    rec_mode = st.selectbox(
        "Modo de recomendação",
        ["Similar a um agente", "Para você (diversificado por clusters)"]
    )

    if rec_mode == "Similar a um agente":
        all_ids = am.list_all_agents()
        if not all_ids:
            st.info("Nenhum agente encontrado.")
            return
        seed = st.selectbox("Escolha um agente de referência", all_ids)
        ids = am.get_similar_agents(seed, k=8)
    else:
        ids = am.get_cluster_representatives(top=12)

    if not ids:
        st.info("Sem recomendações ainda. Peça para executar o job 'reco_index.py'.")
        return

    # Optional tag filter
    cat = get_tag_catalog(bucket, user_email)
    tags = ["(todos)"] + sorted(cat)
    tag_sel = st.selectbox("Filtrar por tag", tags)

    for aid in ids:
        ag = am.get_agent(aid)
        if isinstance(ag, str):
            continue
        tg = am.get_tags(aid)
        if tag_sel != "(todos)" and tag_sel not in tg:
            continue
        with st.container(border=True):
            st.markdown(f"**{ag.nome_agente}**")
            if ag.desc:
                st.write(ag.desc[:220] + ("…" if len(ag.desc) > 220 else ""))
            if tg:
                st.caption("Tags: " + ", ".join(tg))
            st.button("Acessar", key=f"open-{aid}")


def render_tag_editor(bucket: str, user_email: str, agent_id: str | None = None):
    am = _am(bucket, user_email)
    st.subheader("Tag an Agent")
    ids = am.list_all_agents()
    if not ids:
        st.info("Nenhum agente encontrado.")
        return
    aid = st.selectbox("Selecione o agente", ids, index=0 if agent_id is None else max(0, ids.index(agent_id)))
    cur = am.get_tags(aid)
    t = st.text_input("Tags (separe por vírgula)", value=", ".join(cur))
    if st.button("Salvar tags"):
        tags = [x.strip() for x in t.split(",") if x.strip()]
        am.replace_tags(aid, tags)
        st.success("Tags atualizadas!")