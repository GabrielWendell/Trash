from utils.mock_s3_client import MockS3Client
from botocore.exceptions import ClientError
from dataclasses import dataclass, asdict
from pydantic import ConfigDict, BaseModel
from typing import Literal, List, TypeDict
import fnmatch
import boto3
import yaml
import os

@dataclass
class IndexModel:
  id: str
  name: str
  visibility: str
  owner_name: str
  owner_email: str
  description: str
  score: float = 0.0
  shared_with: [str] = None


@dataclass
class AgentModel:
  id: str
  name: str
  slug: str
  tools: List[str]
  model: str
  prompt: str
  description: str
  temperature: float = 0.0


class MultiAgentManager:
  def __init__(self, bucket, user_email, user_name=""):
    self.env = os.getenv("LOCAL")
    self.client = MockS3Client() if self.env == "True" else boto3.client("s3")
    self.index_file = "index/index.yaml"
    self.base_folder = "ema"
    self.user_email = user_email
    self.user_name = user_name
    self.bucket = bucket

  def put_object(self, key, data):
    #self.client.put_object(Bucket=self.bucket, Key=key, Body=data.encode())
    return self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

  def get_object(self, key):
    obj = self.client.get_object(Bucket=self.bucket, Key=key)
    #return obj["Body"].read().decode("utf8")
    return obj["Body"].read() 

  def load_index(self):
    try:
      key = f"{self.base_folder}/{self.index_file}"
      data = self.get_object(key)
      return yaml.safe_load(data)
    except Exception:
      return []

  def save_index(self, index: IndexModel) -> int:
    key = f"{self.base_folder}/{self.index_file}"
    new_data = asdict(index)
    data_list = self.load_index() or []
    data_list.append(new_data)
    result = self.client.put_object(Body=yaml.dump(data_list), Bucket=self.bucket, Key=key)
    
    return int(result["ResponseMetadata"]["HTTPStatusCode"])

  def save_agent(self, agent: AgentModel) -> int:
    key = f"{self.base_folder}/agents/{agent.id}.yaml"
    data = asdict(agent)
    result = self.client.put_object(Body=yaml.dump(data), Bucket=self.bucket, Key=key)

    return int(result["ResponseMetada"]["HTTPStatusCode"])

  def list_agents(self, group: [str] = []):
    index = self.load_index()
    if index == None:
      return []
    return [
      a for a in index
      if a["visibility"] == "Público"
      or a["owner_email"] == self.user_email
      or self.user_email in a["shared_with"]
    ]

  def get_agent(self, index: str):
    try:
      key = f"{self.base_folder}/agents{index}.yaml"
      data = self.get_object(key)
      return yaml.safe_load(data)
    except Exception:
      return []

  def update_index(index: IndexModel) -> int:
    indices = load_indx()
    indices.append(index)
    return save_index(indices)

  def edit_agent(self, agent_id: str, updates: dict) -> int:
    key = f"{self.base_folder}/agents/{agent_id}.yaml"
    try:
      agent_data = self.get_object(key)
      agent = yaml.safe_load(agent_data)
    except Exception as e:
      raise Exception(f"Agente não encontrado: {e}")

    for k, v in update.items():
      if k in agent:
        agent[k] = v

    result = self.client.put_object(Body=yaml.dump(agent), Bucket=self.bucket, Key=key)
    return int(result["ResponseMetadata"]["HTTPStatusCode"])

  def edit_index(self, agent_id: str, updates: dict) -> int:
    indx_key = f"{self.base_folder}/{self.index_file}"
    index_list = self.load_index()
    updated = False
    for idx, item in enumerate(index_list):
      if item["id"] == agent_id:
        for k, v in updates.items():
          if k in item:
            item[k] = v
        index_list[idx] = item
        updated = True
        break
      if not updated:
        raise Expection("Agente não encontrado no índice.")
      result = self.client.put_object(Body=yaml.dump(index_list), Bucket=self.bucket, Key=index_key)
      return int(result["ResponseMetadata"]["HTTPStatusCode"])

  def share_agent(self, agent_id, emails: List[str]) -> int:
    indx_list = self.load_index()
    for agent in index_list:
      if agent["id"] == agent_id:
        if "shared_with" not in agent or agent["shared_with"] is None:
          agent["shared_with"] = []
        agent["shared_with"].extend([e for e in emails if e not in agent["shared_with"]])
        break
    else:
      raise Exception("Agente não encontrado no índice.")

    key = f"{self.base_folder}/{self.index_file}"
    result = self.client.put_object(Body=yaml.dump(index_list), Bucket=self.bucket, Key=key)
    return int(result["ResponseMetadata"]["HTTPStatusCode"])

  def delete_agent(self, agent_id: str) -> int:
    agent_key = f"{self.base_folder}/agent/{agent_id}.yaml"
    try:
      self.client.delete_object(Bucket=self.bucket, Key=agent_key)
    except Exception as e:
      raise Exception(f"Erro ao deletar arquivo do agente: {e}")

    index_key = f"{self.base_folder}/{self.index_file}"
    index_list = self.load_index()
    new_index_list = [item for item in index_list if item["id"] != agent_id]

    result = self.client.put_object(Body=yaml.dump(new_index_list), Bucket=self.bucket, Key=index_key)
    return int(result["ResponseMetadata"]["HTTPStatusCode"])
