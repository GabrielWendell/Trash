# Build payload dicts as before
agent_payload = {
    'id': agent_id, 'name': name, 'slug': slug, 'tools': [],
    'model': model, 'prompt': prompt, 'description': description, 'temperature': temperature,
}
index_payload = {
    'id': agent_id, 'name': name, 'visibility': visibility,
    'owner_name': username, 'owner_email': owner_email,
    'description': description, 'score': (r.get('score') or 0.0) or 0.0,
}

# ↓↓↓ NEW: create real dataclass instances ↓↓↓
try:
    agent_obj = AgentModelCls(**agent_payload)
except TypeError:
    # be permissive if score/temperature types vary etc.
    agent_payload['temperature'] = float(agent_payload.get('temperature') or 0.0)
    agent_obj = AgentModelCls(**agent_payload)

try:
    # ensure score is numeric (many masters carry None/str)
    s = index_payload.get("score")
    index_payload["score"] = float(s) if s not in (None, "", "nan", "NaN") else 0.0
except Exception:
    index_payload["score"] = 0.0
index_obj = IndexModelCls(**index_payload)

# Persist (manager calls dataclasses.asdict(...) inside)
ok_a, res_a = _safe_call('save_agent', agent_obj)
if not ok_a:
    errors += 1
    resultados.append({'name': name, 'owner': owner_email, 'status': 'error',
                       'msg': f'save_agent: {res_a}'})
    prog.progress(min(100, int(i*100/total))); continue

ok_i, res_i = _safe_call('save_index', index_obj)
if not ok_i:
    partial += 1
    resultados.append({'name': name, 'owner': owner_email, 'status': 'partial',
                       'msg': f'save_index: {res_i}', 'agent_id': agent_id})
    prog.progress(min(100, int(i*100/total))); continue
