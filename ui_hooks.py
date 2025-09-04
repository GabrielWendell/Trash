# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List
import streamlit as st
from s3_base import AgentManager

@st.cache_data(ttl=300)
def get_neighbors(bucket: str, user_email: str, agent_id: str, k: int = 8) -> List[str]:
    am = AgentManager(bucket=bucket, user_mail=user_email)
    return am.get_similar_agents(agent_id, k=k)

@st.cache_data(ttl=300)
def get_cluster_representatives(bucket: str, user_email: str, k: int = 12) -> List[str]:
    am = AgentManager(bucket=bucket, user_mail=user_email)
    return am.get_cluster_representatives(top=k)


def render_recommended_section(bucket: str, user_email: str, current_agent_id: str | None = None):
    am = AgentManager(bucket=bucket, user_mail=user_email)
    st.subheader("Recommended Agents")
    if current_agent_id:
        ids = am.get_similar_agents(current_agent_id, k=8)
    else:
        ids = am.get_cluster_representatives(top=12)
    if not ids:
        st.info("No recommendations available yet.")
        return
    for aid in ids:
        ag = am.get_agent(aid)
        if isinstance(ag, str):
            continue
        st.markdown(f"**{ag.nome_agente}** — _{ag.desc[:120]}…_")


def render_tag_editor(bucket: str, user_email: str, agent_id: str):
    am = AgentManager(bucket=bucket, user_mail=user_email)
    cur = am.get_tags(agent_id)
    st.subheader("Tags")
    t = st.text_input("Add comma-separated tags", value=", ".join(cur))
    if st.button("Save Tags"):
        tags = [x.strip() for x in t.split(",") if x.strip()]
        am.replace_tags(agent_id, tags)
        st.success("Tags updated!")