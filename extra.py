# --- add near the top of the branch ---
from dataclasses import make_dataclass, is_dataclass

def _resolve_models():
    """
    Return dataclass classes (AgentModelCls, IndexModelCls).
    If your project exposes AgentModel/IndexModel, use them.
    Otherwise, synthesize compatible dataclasses dynamically.
    """
    AgentModelCls = None
    IndexModelCls = None
    try:
        from create_agent import AgentModel, IndexModel  # your real models
        if is_dataclass(AgentModel):
            AgentModelCls = AgentModel
        if is_dataclass(IndexModel):
            IndexModelCls = IndexModel
    except Exception:
        pass

    if AgentModelCls is None:
        AgentModelCls = make_dataclass(
            "AgentModelDyn",
            [
                ("id", str),
                ("name", str),
                ("slug", str),
                ("tools", list),
                ("model", str),
                ("prompt", str),
                ("description", str),
                ("temperature", float),
            ],
        )
    if IndexModelCls is None:
        IndexModelCls = make_dataclass(
            "IndexModelDyn",
            [
                ("id", str),
                ("name", str),
                ("visibility", str),
                ("owner_name", str),
                ("owner_email", str),
                ("description", str),
                ("score", float, None),  # score can be None; manager typically ignores
            ],
        )
    return AgentModelCls, IndexModelCls

AgentModelCls, IndexModelCls = _resolve_models()
