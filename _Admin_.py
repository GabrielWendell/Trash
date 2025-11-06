elif opcao == 'Criar Agentes (JSON)':
    import json, uuid, math, os, tempfile, traceback
    from dataclasses import make_dataclass, is_dataclass, field
    import streamlit as st

    # resolve manager from session
    mgr = st.session_state.get('multi_agent_manager') or st.session_state.get('agent_manager')

    DEFAULT_MODEL = 'gpt-4.1-2025-04-14'

    # ------------ helpers ------------
    def _clamp(x, lo=0.0, hi=1.0, default=0.0):
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v): return default
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
        for sep in (".", "-", "_", "+"): local = local.replace(sep, " ")
        toks = [t for t in local.split() if t]
        return " ".join(t[0].upper()+t[1:].lower() if len(t)>1 else t.upper() for t in toks)

    def _flatten_shared(shared):
        if isinstance(shared, dict):
            bag = []
            for lst in shared.values():
                if isinstance(lst, list):
                    bag.extend([str(x).strip().lower() for x in lst if str(x).strip()])
            return sorted(set(bag))
        if isinstance(shared, list):
            return sorted(set(str(x).strip().lower() for x in shared if str(x).strip()))
        return []

    def _ensure_str(x, default=""):  # no None to join()
        if isinstance(x, str) and x.strip() != "": return x
        return default

    def _sanitize_agent_payload(p: dict) -> dict:
        p["id"]          = _ensure_str(p.get("id"), default=str(uuid.uuid4()))
        p["name"]        = _ensure_str(p.get("name"), default="Sem Nome")
        p["slug"]        = _ensure_str(p.get("slug"), default="sem-nome")
        p["model"]       = _ensure_str(p.get("model"), default=DEFAULT_MODEL)
        p["prompt"]      = _ensure_str(p.get("prompt"), default="")
        p["description"] = _ensure_str(p.get("description"), default="")
        p["tools"]       = p.get("tools") or []
        p["temperature"] = float(p.get("temperature") or 0.0)
        return p

    def _sanitize_index_payload(p: dict) -> dict:
        p["id"]          = _ensure_str(p.get("id"), default=str(uuid.uuid4()))
        p["name"]        = _ensure_str(p.get("name"), default="Sem Nome")
        p["visibility"]  = _ensure_str(p.get("visibility"), default="Privado")
        p["owner_name"]  = _ensure_str(p.get("owner_name"), default="Usuário")
        p["owner_email"] = _ensure_str(p.get("owner_email"), default="usuario@itau-unibanco.com.br")
        p["description"] = _ensure_str(p.get("description"), default="")
        try:
            s = p.get("score")
            p["score"] = float(s) if s not in (None, "", "nan", "NaN") else 0.0
        except Exception:
            p["score"] = 0.0
        # ensure list
        sw = p.get("shared_with")
        if not isinstance(sw, list):
            p["shared_with"] = []
        return p


    def _safe_call(method_name, *args, **kwargs):
        fn = getattr(mgr, method_name, None)
        if not callable(fn):
            return False, f"método ausente: {method_name}"
        try:
            return True, fn(*args, **kwargs)
        except Exception:
            return False, traceback.format_exc()

    def _resolve_models():
        AgentModelCls = IndexModelCls = None
        try:
            from create_agent import AgentModel, IndexModel  # if available and dataclasses
            if is_dataclass(AgentModel): AgentModelCls = AgentModel
            if is_dataclass(IndexModel): IndexModelCls = IndexModel
        except Exception:
            pass
        if AgentModelCls is None:
            AgentModelCls = make_dataclass("AgentModelDyn", [
                ("id", str), ("name", str), ("slug", str), ("tools", list),
                ("model", str), ("prompt", str), ("description", str), ("temperature", float),
            ])
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
                    ("score", float),
                    ("shared_with", list, field(default_factory=list)),  # <-- NEW
                ],
            )
        return AgentModelCls, IndexModelCls

    def _harden_manager_paths_generic(m):
        """Patch ANY None-valued path-ish attribute so os.path.join never sees None."""
        if m is None: return
        tmp = tempfile.gettempdir()
        for attr in dir(m):
            if attr.startswith("_"):  # skip private
                continue
            try:
                val = getattr(m, attr)
            except Exception:
                continue
            # path-like attributes we sanitize
            name_l = attr.lower()
            if any(tok in name_l for tok in ("dir", "path", "folder", "prefix", "root", "base")):
                if val is None:
                    # directories/paths -> tmp; s3-like prefixes/folders -> safe strings
                    if any(tok in name_l for tok in ("dir", "path", "root")):
                        setattr(m, attr, tmp)
                    else:
                        setattr(m, attr, "Agents_Chatbots")
        # ALSO ensure mgr.bucket is a string if the code happens to join it (rare)
        if getattr(m, "bucket", None) is None:
            setattr(m, "bucket", "dummy-bucket")  # not used for local join; harmless

    # ------------ UI ------------
    st.subheader('Criar novos agentes a partir de um arquivo JSON')
    up = st.file_uploader('Envie o master JSON **filtrado**', type=['json'])
    col1, col2 = st.columns([1,1])
    with col1: dry_run = st.checkbox('Dry run (não salvar)', value=True)
    with col2: go = st.button('Processar JSON e criar agentes', type='primary')

    if go:
        if up is None:
            st.warning('Envie um arquivo JSON antes de processar.'); st.stop()
        try:
            data = json.load(up); assert isinstance(data, list)
        except Exception as e:
            st.error(f'JSON inválido: {e}'); st.stop()
        if mgr is None:
            st.error("Gerenciador não encontrado em st.session_state."); st.stop()

        # 🔧 Make sure the manager has no None path attributes
        _harden_manager_paths_generic(mgr)

        AgentModelCls, IndexModelCls = _resolve_models()

        created = skipped = errors = partial = 0
        resultados, total = [], len(data)
        prog = st.progress(0)

        for i, r in enumerate(data, start=1):
            name        = (r.get('name') or '').strip()
            owner_email = (r.get('owner') or '').strip().lower()
            prompt      = (r.get('prompt') or '').strip()
            model       = (r.get('model') or '').strip() or DEFAULT_MODEL

            missing = []
            if not name:        missing.append('name')
            if not owner_email: missing.append('owner')
            if not prompt:      missing.append('prompt')
            if missing:
                errors += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'error',
                                   'msg': 'campos obrigatórios ausentes: ' + ", ".join(missing)})
                prog.progress(min(100, int(i*100/total))); continue

            agent_id    = (r.get('id') or '').strip() or str(uuid.uuid4())
            description = (r.get('description') or '').strip()
            temperature = _clamp(r.get('temperature'), 0.0, 1.0, 0.0)
            visibility  = r.get('visibility') or 'Privado'
            username    = (r.get('username') or _email_to_username(owner_email))
            shared_with = _flatten_shared(r.get('shared_with', {}))
            slug        = _slugify(name)

            if dry_run:
                created += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'ok', 'msg': 'dry-run'})
                prog.progress(min(100, int(i*100/total))); continue

            agent_payload = _sanitize_agent_payload({
                'id': agent_id, 
                'name': name, 
                'slug': slug, 
                'tools': [],
                'model': model, 
                'prompt': prompt, 
                'description': description, 
                'temperature': temperature,
            })
            index_payload = _sanitize_index_payload({
                "id": agent_id,
                "name": name,
                "visibility": visibility,
                "owner_name": username,
                "owner_email": owner_email,
                "description": description,
                "score": r.get("score"),
                "shared_with": shared_with,            # <-- NEW
            })


            # build dataclasses (manager calls dataclasses.asdict inside)
            try:
                agent_obj = AgentModelCls(**agent_payload)
            except TypeError:
                agent_payload['temperature'] = float(agent_payload.get('temperature') or 0.0)
                agent_obj = AgentModelCls(**agent_payload)
            index_obj = IndexModelCls(**index_payload)

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

            if shared_with: _safe_call('share_agent', agent_id, shared_with)

            created += 1
            resultados.append({'name': name, 'owner': owner_email, 'status': 'ok',
                               'msg': '', 'agent_id': agent_id})
            prog.progress(min(100, int(i*100/total)))

        st.success(f'Concluído: criados={created}, pulados={skipped}, parciais={partial}, erros={errors}')
        st.dataframe(resultados)
