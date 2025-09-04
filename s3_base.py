# -*- coding: utf-8 -*-
"""
EVA S3 helpers — patched to be robust and to support:
  - tags read/write and inverted index
  - fetching precomputed recommendations (neighbors) and cluster terms
  - bug fixes in list operations and key handling

Bucket layout (prefix self.base_folder = 'agents/'):
  agents/
    private/<email>/*.yaml
    public/<email>/*.yaml
    groups/<group_id>/group_agents/*.yaml
    index/  # artifacts produced by reco_index.py (neighbors, clusters, etc.)
      neighbors/<id>.json
      cluster_terms.json
      cluster_representatives.json
    tags/
      _catalog.json
      <tag>.json            # list of agent ids having this tag

This module remains free of heavy ML logic; that is handled by reco_index.py and tags_engine.py.
"""
from __future__ import annotations

import os
import io
import json
import re
import glob
import fnmatch
from typing import List, Dict, Optional, Literal, Tuple

import boto3
from botocore.exceptions import ClientError
import yaml
from pydantic import ConfigDict, BaseModel, Field

# ------------------------------
# Data Models
# ------------------------------
class AgentDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id_agente: str
    nome_agente: str
    desc: str
    msg_inicial: str
    prompt: str
    temp: float = 0.5
    # NEW (optional)
    tags: Optional[List[str]] = Field(default=None)

    def get_dict(self) -> Dict:
        d = {
            "id_agente": self.id_agente,
            "nome_agente": self.nome_agente,
            "desc": self.desc,
            "msg_inicial": self.msg_inicial,
            "prompt": self.prompt,
            "temp": self.temp,
        }
        if self.tags is not None:
            d["tags"] = sorted(set(self.tags))
        return d

    def get_yaml(self) -> str:
        return yaml.safe_dump(self.get_dict(), allow_unicode=True, sort_keys=False)


class GroupDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id_grupo: str
    agents: List[str]
    owner: str
    members: List[str]

    def get_dict(self) -> Dict:
        return {
            "id_grupo": self.id_grupo,
            "agents": self.agents,
            "owner": self.owner,
            "members": self.members,
        }

    def get_yaml(self) -> str:
        return yaml.safe_dump(self.get_dict(), allow_unicode=True, sort_keys=False)


# ------------------------------
# Managers
# ------------------------------
class GroupManager:
    def __init__(self, bucket: str, user_mail: str):
        self.env = os.getenv("LOCAL")
        self.client = boto3.client("s3")
        self.bucket = bucket
        self.base_folder = "agents/groups/"
        self.user_email = user_mail

    # ---- group CRUD ----
    def create_new_group(self, group: GroupDefinition):
        if group.id_grupo in self.list_all_group_ids():
            return "duplicate"
        key = f"{self.base_folder}{group.id_grupo}/{group.id_grupo}.yaml"
        res = self.client.put_object(Body=group.get_yaml().encode("utf8"), Bucket=self.bucket, Key=key)
        return int(res["ResponseMetadata"]["HTTPStatusCode"])

    def get_group_by_id(self, group_id: str) -> Dict | str:
        key = f"{self.base_folder}{group_id}/{group_id}.yaml"
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            y = obj["Body"].read().decode("utf8")
            return yaml.safe_load(y)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            return "Grupo não encontrado" if code == "NoSuchKey" else str(code)

    def list_all_group_ids(self) -> List[str]:
        prefix = self.base_folder
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        paths = [c["Key"] for c in resp.get("Contents", []) if c["Key"].endswith(".yaml")]
        return [os.path.basename(p).removesuffix(".yaml") for p in paths]

    def get_all_groups(self) -> List[Dict]:
        ids = self.list_all_group_ids()
        out = []
        for gid in ids:
            g = self.get_group_by_id(gid)
            if isinstance(g, dict):
                out.append(g)
        return out

    def list_user_owned_groups(self) -> List[Dict]:
        return [g for g in self.get_all_groups() if g.get("owner") == self.user_email]

    def list_user_accessible_groups(self) -> List[Dict]:
        groups = self.get_all_groups()
        allowed = [g for g in groups if self.user_email in (g.get("members") or [])]
        return self.list_user_owned_groups() + allowed

    def edit_group(self, id_grupo: str, agents: Optional[List[str]] = None, members: Optional[List[str]] = None):
        if id_grupo not in self.list_all_group_ids():
            return "group not found"
        g = self.get_group_by_id(id_grupo)
        if not isinstance(g, dict):
            return g
        if agents is not None:
            g["agents"] = list(dict.fromkeys(agents))
        if members is not None:
            g["members"] = list(dict.fromkeys(members))
        key = f"{self.base_folder}{id_grupo}/{id_grupo}.yaml"
        res = self.client.put_object(Body=GroupDefinition(**g).get_yaml().encode("utf8"), Bucket=self.bucket, Key=key)
        return int(res["ResponseMetadata"]["HTTPStatusCode"])


class AgentManager:
    def __init__(self, bucket: str, user_mail: str):
        self.env = os.getenv("LOCAL")
        self.client = boto3.client("s3")
        self.bucket = bucket
        self.base_folder = "agents/"
        self.user_email = user_mail
        self.group_manager = GroupManager(bucket, user_mail)
        self.initialization()

    # ---------- utils ----------
    def _s3_list(self, prefix: str, suffix: Optional[str] = None) -> List[str]:
        """List *all* keys under prefix (handles pagination)."""
        token = None
        out: List[str] = []
        while True:
            kwargs = dict(Bucket=self.bucket, Prefix=prefix)
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            for c in resp.get("Contents", []):
                k = c["Key"]
                if suffix is None or k.endswith(suffix):
                    out.append(k)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return out

    def _get_yaml(self, key: str) -> Optional[Dict]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            y = obj["Body"].read().decode("utf8")
            return yaml.safe_load(y)
        except ClientError:
            return None

    def _put_yaml(self, key: str, data: Dict) -> int:
        res = self.client.put_object(Body=yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode("utf8"),
                                     Bucket=self.bucket, Key=key)
        return int(res["ResponseMetadata"]["HTTPStatusCode"])

    def _key_for_agent(self, id_agente: str) -> Optional[str]:
        # search in private, public, and groups
        for prefix in ["private/", "public/", "groups/"]:
            keys = self._s3_list(self.base_folder + prefix)
            for k in keys:
                if k.endswith(f"/{id_agente}.yaml"):
                    return k
        return None

    # ---------- basics ----------
    def initialization(self):
        pass

    def is_public(self, agent_id: str) -> bool:
        keys = self._s3_list(self.base_folder + "public/", suffix=f"{agent_id}.yaml")
        return any(k.endswith(f"/{agent_id}.yaml") for k in keys)

    def is_built_in(self, agent_id: str) -> bool:
        return agent_id in self.list_built_in_agents()

    def get_agent_owner(self, agent_id: str) -> str:
        k = self._key_for_agent(agent_id)
        if not k:
            return ""
        # agents/<scope>/<owner>/... or agents/groups/<gid>/group_agents/...
        parts = k[len(self.base_folder):].split("/")
        if parts[0] == "groups":
            gid = parts[1]
            g = self.group_manager.get_group_by_id(gid)
            return (g or {}).get("owner", "") if isinstance(g, dict) else ""
        if self.is_built_in(agent_id):
            return "Equipe EVA"
        # scope/owner/file
        return parts[1] if len(parts) >= 3 else ""

    def list_built_in_agents(self) -> List[str]:
        return [os.path.basename(p).removesuffix(".yaml") for p in glob.glob("src/static/agents/*.yaml")]

    def list_group_agents(self, group_id: str) -> List[str]:
        prefix = f"{self.base_folder}groups/{group_id}/group_agents/"
        keys = self._s3_list(prefix, suffix=".yaml")
        return [os.path.basename(k).removesuffix(".yaml") for k in keys]

    def list_user_agents(self, fetch_public: bool = True) -> List[str]:
        paths: List[str] = []
        # private
        prefix = f"{self.base_folder}private/{self.user_email}/"
        paths += self._s3_list(prefix, suffix=".yaml")
        # public (optional)
        if fetch_public:
            paths += self._s3_list(self.base_folder + "public/", suffix=".yaml")
        # built-ins
        ids = [p.split("/")[-1].removesuffix(".yaml") for p in paths]
        ids += self.list_built_in_agents()
        return sorted(list(dict.fromkeys(ids)))

    def list_all_agents(self) -> List[str]:
        paths = self._s3_list(self.base_folder, suffix=".yaml")
        ids = [os.path.basename(p).removesuffix(".yaml") for p in paths]
        ids += self.list_built_in_agents()
        return sorted(list(dict.fromkeys(ids)))

    def get_agent(self, id_agente: str, agent_group: Optional[str] = None) -> AgentDefinition | str:
        if self.is_built_in(id_agente):
            with open(f"src/static/agents/{id_agente}.yaml", encoding="utf8") as f:
                return AgentDefinition(**yaml.safe_load(f.read()))
        # S3 resolution
        k = self._key_for_agent(id_agente)
        if not k and agent_group:
            k = f"{self.base_folder}groups/{agent_group}/group_agents/{id_agente}.yaml"
        if not k:
            return "Agente não encontrado"
        data = self._get_yaml(k)
        return AgentDefinition(**data) if isinstance(data, dict) else "Agente não encontrado"

    def save_agent(self, agent: AgentDefinition) -> int:
        key = f"{self.base_folder}private/{self.user_email}/{agent.id_agente}.yaml"
        return self._put_yaml(key, agent.get_dict())

    def update_agent_in_all_locations(self, agent: AgentDefinition) -> int:
        code = 200
        # private
        key_priv = f"{self.base_folder}private/{self.user_email}/{agent.id_agente}.yaml"
        code = self._put_yaml(key_priv, agent.get_dict())
        # public (if it exists there)
        if self.is_public(agent.id_agente):
            owner = self.get_agent_owner(agent.id_agente)
            key_pub = f"{self.base_folder}public/{owner}/{agent.id_agente}.yaml"
            code = self._put_yaml(key_pub, agent.get_dict())
        # groups
        for g in self.group_manager.get_all_groups():
            if agent.id_agente in (g.get("agents") or []):
                key_grp = f"{self.base_folder}groups/{g['id_grupo']}/group_agents/{agent.id_agente}.yaml"
                code = self._put_yaml(key_grp, agent.get_dict())
        return code

    def delete_private_agent(self, id_agente: str) -> int:
        key = f"{self.base_folder}private/{self.user_email}/{id_agente}.yaml"
        res = self.client.delete_object(Bucket=self.bucket, Key=key)
        return int(res["ResponseMetadata"]["HTTPStatusCode"])

    def delete_group_agent(self, id_agente: str, id_grupo: str):
        try:
            info = self.group_manager.get_group_by_id(id_grupo)
        except Exception:
            info = None
        if not isinstance(info, dict):
            # also attempt to delete public copy (if any)
            if self.is_public(id_agente):
                owner = self.get_agent_owner(id_agente)
                key_pub = f"{self.base_folder}public/{owner}/{id_agente}.yaml"
                self.client.delete_object(Bucket=self.bucket, Key=key_pub)
            return "Agente excluído com sucesso"
        if id_agente not in (info.get("agents") or []):
            return "Agente não pertence ao grupo"
        key = f"{self.base_folder}groups/{id_grupo}/group_agents/{id_agente}.yaml"
        res = self.client.delete_object(Bucket=self.bucket, Key=key)
        if res["ResponseMetadata"]["HTTPStatusCode"] in (200, 204):
            info["agents"].remove(id_agente)
            self.group_manager.edit_group(id_grupo, agents=info["agents"])
            if self.is_public(id_agente):
                owner = self.get_agent_owner(id_agente)
                key_pub = f"{self.base_folder}public/{owner}/{id_agente}.yaml"
                self.client.delete_object(Bucket=self.bucket, Key=key_pub)
            return "Agente excluído com sucesso"
        return "Falha ao excluir agente"

    def move_agent(self, id_agente: str, move_target: Literal["group", "public"], group_id: Optional[str] = None):
        agent = self.get_agent(id_agente, group_id if move_target == "public" else None)
        if isinstance(agent, str):
            return agent
        y = agent.get_yaml().encode("utf8")
        if move_target == "group" and group_id:
            g = self.group_manager.get_group_by_id(group_id)
            new_key = f"{self.base_folder}groups/{group_id}/group_agents/{id_agente}.yaml"
            g_agents = (g.get("agents") or []) + [id_agente]
            self.group_manager.edit_group(group_id, agents=g_agents)
        else:
            owner = self.get_agent_owner(id_agente) or self.user_email
            new_key = f"{self.base_folder}{move_target}/{owner}/{id_agente}.yaml"
        res = self.client.put_object(Body=y, Bucket=self.bucket, Key=new_key)
        return int(res["ResponseMetadata"]["HTTPStatusCode"])

    # ---------- TAGS support ----------
    def add_tags(self, id_agente: str, tags: List[str]) -> int:
        key = self._key_for_agent(id_agente)
        if not key:
            return 404
        data = self._get_yaml(key) or {}
        cur = set((data.get("tags") or []))
        new = sorted(set(t.strip().lower() for t in tags if t and isinstance(t, str)))
        data["tags"] = sorted(cur.union(new))
        code = self._put_yaml(key, data)
        if code in (200, 204):
            self._update_tag_index(id_agente, before=list(cur), after=data["tags"])
        return code

    def replace_tags(self, id_agente: str, tags: List[str]) -> int:
        key = self._key_for_agent(id_agente)
        if not key:
            return 404
        data = self._get_yaml(key) or {}
        before = data.get("tags") or []
        data["tags"] = sorted(set(t.strip().lower() for t in tags if t))
        code = self._put_yaml(key, data)
        if code in (200, 204):
            self._update_tag_index(id_agente, before=before, after=data["tags"])
        return code

    def get_tags(self, id_agente: str) -> List[str]:
        key = self._key_for_agent(id_agente)
        if not key:
            return []
        data = self._get_yaml(key) or {}
        return sorted(set(data.get("tags") or []))

    def list_agents_by_tag(self, tag: str) -> List[str]:
        key = f"{self.base_folder}tags/{tag.lower()}.json"
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(obj["Body"].read())
        except ClientError:
            return []

    def _update_tag_index(self, id_agente: str, before: List[str], after: List[str]):
        removed = set(before) - set(after)
        added = set(after) - set(before)
        for t in removed:
            self._remove_from_tag(t, id_agente)
        for t in added:
            self._add_to_tag(t, id_agente)

    def _add_to_tag(self, tag: str, id_agente: str):
        tag = tag.lower()
        key = f"{self.base_folder}tags/{tag}.json"
        lst = self.list_agents_by_tag(tag)
        if id_agente not in lst:
            lst.append(id_agente)
            self.client.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(sorted(lst)).encode("utf8"))
        self._update_catalog()

    def _remove_from_tag(self, tag: str, id_agente: str):
        tag = tag.lower()
        key = f"{self.base_folder}tags/{tag}.json"
        lst = [x for x in self.list_agents_by_tag(tag) if x != id_agente]
        self.client.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(sorted(lst)).encode("utf8"))
        self._update_catalog()

    def _update_catalog(self):
        prefix = f"{self.base_folder}tags/"
        keys = [k for k in self._s3_list(prefix, suffix=".json") if not k.endswith("_catalog.json")]
        cat = {}
        for k in keys:
            tag = os.path.basename(k).removesuffix('.json')
            try:
                obj = self.client.get_object(Bucket=self.bucket, Key=k)
                cat[tag] = len(json.loads(obj["Body"].read()))
            except ClientError:
                cat[tag] = 0
        self.client.put_object(Bucket=self.bucket, Key=prefix+"_catalog.json", Body=json.dumps(cat, ensure_ascii=False).encode("utf8"))

    # ---------- RECOMMENDATIONS (serving precomputed) ----------
    def get_similar_agents(self, id_agente: str, k: int = 8) -> List[str]:
        """Return up to k similar agents using the offline neighbor index."""
        key = f"{self.base_folder}index/neighbors/{id_agente}.json"
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            lst = json.loads(obj["Body"].read())
            return lst[:k]
        except ClientError:
            return []

    def get_cluster_terms(self) -> Dict[str, List[str]]:
        key = f"{self.base_folder}index/cluster_terms.json"
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(obj["Body"].read())
        except ClientError:
            return {}

    def get_cluster_representatives(self, top: int = 12) -> List[str]:
        key = f"{self.base_folder}index/cluster_representatives.json"
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
            reps = json.loads(obj["Body"].read())
            return reps[:top]
        except ClientError:
            return []