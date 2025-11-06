# ---- add these helpers near the top of your Admin branch ----
import os, tempfile

def _ensure_str(x, default=""):
    if isinstance(x, str) and x.strip() != "":
        return x
    return default

def _harden_manager_paths(mgr):
    """
    Ensure manager path attributes used by save_agent/save_index are strings.
    We don't modify your code; just patch attributes at runtime if they're None.
    """
    # Common names used in similar managers; only set when missing or falsy.
    defaults = {
        "base_dir": tempfile.gettempdir(),      # local scratch
        "local_dir": tempfile.gettempdir(),     # some code uses 'local_dir'
        "root_dir": tempfile.gettempdir(),      # or 'root_dir'
        "base_folder": "Agents_Chatbots",       # s3 logical root used in keys
        "bucket_folder": "Agents_Chatbots",     # some variants use 'bucket_folder'
        "agents_prefix": "agents",              # prefix for agent files
        "index_prefix": "index",                # prefix for index files
    }
    for attr, val in defaults.items():
        cur = getattr(mgr, attr, None)
        if not isinstance(cur, str) or cur.strip() == "":
            setattr(mgr, attr, val)

def _sanitize_agent_payload(payload: dict) -> dict:
    # make sure no field that might be used in path formatting is None
    payload["id"] = _ensure_str(payload.get("id"), default=str(uuid.uuid4()))
    payload["name"] = _ensure_str(payload.get("name"), default="Sem Nome")
    payload["slug"] = _ensure_str(payload.get("slug"), default="sem-nome")
    payload["model"] = _ensure_str(payload.get("model"), default=DEFAULT_MODEL)
    payload["prompt"] = _ensure_str(payload.get("prompt"), default="")
    payload["description"] = _ensure_str(payload.get("description"), default="")
    payload["tools"] = payload.get("tools") or []          # list, not None
    payload["temperature"] = float(payload.get("temperature") or 0.0)
    return payload

def _sanitize_index_payload(payload: dict) -> dict:
    payload["id"] = _ensure_str(payload.get("id"), default=str(uuid.uuid4()))
    payload["name"] = _ensure_str(payload.get("name"), default="Sem Nome")
    payload["visibility"] = _ensure_str(payload.get("visibility"), default="Privado")
    payload["owner_name"] = _ensure_str(payload.get("owner_name"), default="Usuário")
    payload["owner_email"] = _ensure_str(payload.get("owner_email"), default="usuario@itau-unibanco.com.br")
    payload["description"] = _ensure_str(payload.get("description"), default="")
    try:
        s = payload.get("score")
        payload["score"] = float(s) if s not in (None, "", "nan", "NaN") else 0.0
    except Exception:
        payload["score"] = 0.0
    return payload

# ---------------------

# Then, right before the loop (after you resolve mgr) add:
# Harden manager attrs so os.path.join(...) never sees None
_harden_manager_paths(mgr)

# ---------------------

# And where you build the objects, sanitize the payloads before creating the dataclass instances:
agent_payload = _sanitize_agent_payload({
    'id': agent_id, 'name': name, 'slug': slug, 'tools': [],
    'model': model, 'prompt': prompt, 'description': description, 'temperature': temperature,
})

index_payload = _sanitize_index_payload({
    'id': agent_id, 'name': name, 'visibility': visibility,
    'owner_name': username, 'owner_email': owner_email,
    'description': description, 'score': r.get('score'),
})

# ---------------------

# Keep the dataclass construction from the previous step:
from dataclasses import make_dataclass, is_dataclass
# ... AgentModelCls, IndexModelCls = _resolve_models() ...

agent_obj = AgentModelCls(**agent_payload)
index_obj = IndexModelCls(**index_payload)
