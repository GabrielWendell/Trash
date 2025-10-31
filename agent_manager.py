from langgraph.checkpoint.memory import MemorySaver
from ema.core.s3_agent_manager import AgentModel
from ema.core.base_agents import BaseAgent
from ema.core.chat_iara import ChatIaraGenIA
from ema.core.memory import memory
from typing import List

def create_agent(agent: AgentModel, memory: MemorySaver = None):
  llm = ChatIaraGenIA(agent["model"], agent["temperature"])

  return BaseAgent(
        model = llm,
        tools = agent["tools"],
        name = agent["slug"],
        description = agent["description"],
        checkpointer = memory,
        prompt = agent["prompt"]
      )
