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
    shared_with = st.text_input(label="Compartilhar com [lista de e-mails separados por vírdula]")

    visibility_col, model_col, tools_col, temp_col = st.columns([0.2, 0.2, 0.3, 0.3])

    with visibility_col:
      visibility = st.radio(key="visibility", label="Visibilidade", options=["Privado", "Público"])

    with model_col:
      model = st.selectbox(label="Selecione um modelo", options=models, placeholder="Selecione um modelo", key="model")

    with tools_col:
      tools = st.multiselect("Selecione tools para o Agente", options=tools_available, default=[])
      tools_ids="".join(map(str,tools))

    with temp_col:
      temperature = st.slider(label="Temperature (quanto maior, maior a criatividade!):", min_value = 0.0, max_value = 1.0, value = 0.0, key = "temperature")

    submit_button = st.form_submit_button(label="Criar", type="primary")

    if submit_button:
      if agent_name == "" or agent_description == "" or prompt == "" or model == "":
        st.error("Todos os campos precisam ser preenchidos", icon="*")
      else:
        agent_manager = st.sesstion_state["multi_agent_manager"]
        agent_id = str(uuid.uuid4())
        def_agent = {
            "id": agent_id,
            "name": agent_name,
            "slug": slugify(agent_name),
            "model": model,
            "tools": [],
            "prompt": prompt,
            "description": agent_description,
            "temperature": temperature,
          }
        index_agent = {
            "id": agent_id,
            "name": agent_name,
            "owner_name": agent_manager.user_name,
            "owner_email": agent_manager.user_email,
            "visibility": visibility,
            "description": description,
            "shared_with": shared_with,   
          }

        agent = AgentModel(**def_agent)
        index = IndexModel(**index_agent)

        response_agent = agent_manager.save_agent(agent)
        response_index = agent_manager.save_index(index)

  if (response_index == 200) and (response_agent == 200):
    st.success(f"Agente '{agent_name}' salvo com sucesso!")
    st.session_state.list_index_agents.append(index_agent)
    st.session_state.agent_cards.append({
                "id": index_agent["id"],
                "name": index_agent["name"],
                "viewer": agent_manager.user_email,
                "owner_name": index_agent["owner_name"],
                "owner_email": index_agent["owner_email"],
                "description": index_agent["description"],
              })
    
    st.rerun()
  
