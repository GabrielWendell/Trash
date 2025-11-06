# ===================== Criar Agentes (JSON) — FINAL FIX =====================
# This version works around the 'dict has no attribute id' issue.
# It dynamically converts dicts into lightweight objects with attribute access.

elif opcao == 'Criar Agentes (JSON)':
    from dataclasses import make_dataclass, is_dataclass
    import json, uuid, math, types
    import streamlit as st
    import os, tempfile

    try:
        mgr = st.session_state.get('multi_agent_manager') or st.session_state.get('agent_manager')
    except Exception:
        mgr = None

    DEFAULT_MODEL = 'gpt-4.1-2025-04-14'

    def _clamp(x, lo=0.0, hi=1.0, default=0.0):
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return default
            return max(lo, min(hi, v))
        except Exception:
            return default

    def _slugify(name: str) -> str:
        s = (name or '').strip()
        for ch in " /\t\n\r\f\v\\":
            s = s.replace(ch, " ")
        return "-".join(s.split()).lower()

    def _email_to_username(email: str) -> str:
        local = (email or "").split("@", 1)[0]
        for sep in (".", "-", "_", "+"):
            local = local.replace(sep, " ")
        toks = [t for t in local.split() if t]
        return " ".join(t[0].upper()+t[1:].lower() if len(t)>1 else t.upper() for t in toks)

    def _flatten_shared(shared):
        if isinstance(shared, dict):
            bag = []
            for _, lst in shared.items():
                if isinstance(lst, list):
                    bag.extend([str(x).strip().lower() for x in lst if str(x).strip()])
            return sorted(set(bag))
        if isinstance(shared, list):
            return sorted(set(str(x).strip().lower() for x in shared if str(x).strip()))
        return []

    def _to_object(d: dict):
        """Convert dict to a simple object with attribute access."""
        o = types.SimpleNamespace()
        for k, v in d.items():
            setattr(o, k, v)
        return o

    def _safe_call(method_name, *args, **kwargs):
        fn = getattr(mgr, method_name, None)
        if not callable(fn):
            return False, f"método ausente: {method_name}"
        try:
            return True, fn(*args, **kwargs)
        except Exception as e:
            return False, str(e)

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



    st.subheader('Criar novos agentes a partir de um arquivo JSON')
    up = st.file_uploader('Envie o master JSON **filtrado**', type=['json'])
    col1, col2 = st.columns([1,1])
    with col1:
        dry_run = st.checkbox('Dry run (não salvar)', value=True)
    with col2:
        go = st.button('Processar JSON e criar agentes', type='primary')

    if go:
        if up is None:
            st.warning('Envie um arquivo JSON antes de processar.')
            st.stop()
        try:
            data = json.load(up)
            assert isinstance(data, list)
        except Exception as e:
            st.error(f'JSON inválido: {e}')
            st.stop()

        if mgr is None:
            st.error("Gerenciador não encontrado em st.session_state.")
            st.stop()

        # Harden manager attrs so os.path.join(...) never sees None
        _harden_manager_paths(mgr)

        created = skipped = errors = partial = 0
        resultados = []
        total = len(data)
        prog = st.progress(0)

        for i, r in enumerate(data, start=1):
            name = (r.get('name') or '').strip()
            owner_email = (r.get('owner') or '').strip().lower()
            prompt = (r.get('prompt') or '').strip()
            model = (r.get('model') or '').strip() or DEFAULT_MODEL

            missing = []
            if not name:        missing.append('name')
            if not owner_email: missing.append('owner')
            if not prompt:      missing.append('prompt')

            if missing:
                errors += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'error',
                                   'msg': 'campos obrigatórios ausentes: ' + ", ".join(missing)})
                prog.progress(min(100, int(i*100/total))); continue

            agent_id = (r.get('id') or '').strip() or str(uuid.uuid4())
            description  = (r.get('description') or '').strip()
            temperature  = _clamp(r.get('temperature'), 0.0, 1.0, 0.0)
            visibility   = r.get('visibility') or 'Privado'
            username     = (r.get('username') or _email_to_username(owner_email))
            shared_with  = _flatten_shared(r.get('shared_with', {}))
            slug         = _slugify(name)

            if dry_run:
                created += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'ok', 'msg': 'dry-run'})
                prog.progress(min(100, int(i*100/total))); continue

            # Build payloads
            agent_payload = _sanitize_agent_payload({
                'id': agent_id, 'name': name, 'slug': slug, 'tools': [],
                'model': model, 'prompt': prompt, 'description': description, 'temperature': temperature,
            })
            
            index_payload = _sanitize_index_payload({
                'id': agent_id, 'name': name, 'visibility': visibility,
                'owner_name': username, 'owner_email': owner_email,
                'description': description, 'score': r.get('score'),
            })

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

            # Convert dicts to simple objects for mgr.save_agent compatibility
            agent_obj = _to_object(agent_payload)
            index_obj = _to_object(index_payload)

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

            if shared_with:
                _safe_call('share_agent', agent_id, shared_with)

            created += 1
            resultados.append({'name': name, 'owner': owner_email, 'status': 'ok', 'msg': '', 'agent_id': agent_id})
            prog.progress(min(100, int(i*100/total)))

        st.success(f'Concluído: criados={created}, pulados={skipped}, parciais={partial}, erros={errors}')
        st.dataframe(resultados)
# ===================== /Criar Agentes (JSON) — FINAL FIX =====================
