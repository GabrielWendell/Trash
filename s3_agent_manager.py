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
