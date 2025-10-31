from ema.core.s3_agent_manager import MultiAgentManager, AgentModel, IndexModel
from ema.core.utils import slugify
import streamlit as st
import uuid

def create_agent():
  tools_available = []
  models = ["", "gpt-4.1-2025-04-14"]
  resposnse_agent = 0
  response_index = 0

with st.form(key="create_agent_form", clear_on_submit=True):
  agent_name = st.text_input(label="Nome do Agente")
  agent_description = st.text_area(label="Descrição do Agente")
  prompt = st.text_area(label="Prompt")
  
