# --- IMPORTS (adicione no topo do Admin.py, se ainda não existirem) ---
from ema.core.s3_agent_manager import MultiAgentManager, AgentModel, IndexModel  # type: ignore
from ema.core.utils import slugify  # type: ignore
import streamlit as st
import json, uuid, math

# --- HELPER(S) ---
def _clamp(x, lo=0.0, hi=1.0, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return max(lo, min(hi, v))
    except Exception:
        return default

# >>>>>>>>>>>>>>>>>>>>>  BLOCO DO MENU  <<<<<<<<<<<<<<<<<<<<<<
# 1) Acrescente a opção no selectbox existente
opcao = st.selectbox('Escolha uma opção:', ('', 'Dashboard', 'Logs', 'Feedback', 'Limpeza de Logs', 'Criar Agentes (JSON)'))

# 2) Novo ramo do menu
# ===================== Criar Agentes (JSON) — FIXED BRANCH =====================
elif opcao == 'Criar Agentes (JSON)':
    import json, uuid, math, time
    import streamlit as st

    # OPTIONAL: import pydantic models if available (falls back to dicts)
    try:
        from create_agent import AgentModel  # type: ignore
    except Exception:
        AgentModel = None  # type: ignore
    try:
        # use whatever manager you put in st.session_state (no file edits)
        mgr = st.session_state.get('multi_agent_manager') or st.session_state.get('agent_manager')
    except Exception:
        mgr = None

    # --- helpers (local to this branch) ---
    DEFAULT_MODEL = 'gpt-4.1-2025-04-14'  # safe fallback if model is null/empty

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

    def _safe_call(method_name, *args, **kwargs):
        fn = getattr(mgr, method_name, None)
        if not callable(fn):
            return False, f"método ausente: {method_name}"
        try:
            return True, fn(*args, **kwargs)
        except Exception as e:
            return False, str(e)

    def _already_exists(owner_email: str, slug: str, agent_id: str | None) -> bool:
        # If ID present, try direct fetch
        if agent_id:
            ok, res = _safe_call("get_agent", agent_id)
            if ok and res:
                return True
        # Otherwise scan index using any available lister
        for m in ("list_indexes", "load_index", "get_indexes", "get_index"):
            ok, res = _safe_call(m)
            if ok and res:
                items = res if isinstance(res, list) else res.get("items", [])
                try:
                    for it in items:
                        it_owner = str(it.get("owner_email", "")).lower()
                        it_slug = _slugify(str(it.get("name", "")))
                        if it_owner == owner_email and it_slug == slug:
                            return True
                except Exception:
                    pass
        return False

    # --- UI ---
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
            st.error("Gerenciador não encontrado em st.session_state (ex.: 'multi_agent_manager').")
            st.stop()

        created = skipped = errors = partial = 0
        resultados = []
        total = len(data)
        prog = st.progress(0)

        for i, r in enumerate(data, start=1):
            name = (r.get('name') or '').strip()
            owner_email = (r.get('owner') or '').strip().lower()
            prompt = (r.get('prompt') or '').strip()
            # model is OPTIONAL now; default if empty
            model = (r.get('model') or '').strip() or DEFAULT_MODEL

            missing = []
            if not name:        missing.append('name')
            if not owner_email: missing.append('owner')
            if not prompt:      missing.append('prompt')
            # (model no longer required)

            if missing:
                errors += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'error',
                                   'msg': 'campos obrigatórios ausentes: ' + ", ".join(missing)})
                prog.progress(min(100, int(i*100/total))); continue

            agent_id = (r.get('id') or '').strip() or str(uuid.uuid4())
            description  = (r.get('description') or '').strip()
            temperature  = _clamp(r.get('temperature'), 0.0, 1.0, 0.0)
            visibility   = r.get('visibility') or 'Privado'  # já PT-BR
            username     = (r.get('username') or _email_to_username(owner_email))
            shared_with  = _flatten_shared(r.get('shared_with', {}))
            slug         = _slugify(name)

            # Idempotência
            if _already_exists(owner_email, slug, agent_id if agent_id else None):
                skipped += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'skip', 'msg': 'já existe'})
                prog.progress(min(100, int(i*100/total))); continue

            # Preview only
            if dry_run:
                created += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'ok', 'msg': 'dry-run'})
                prog.progress(min(100, int(i*100/total))); continue

            # Build payloads; if pydantic models exist, use them; else pass dicts.
            agent_payload = {
                'id': agent_id, 'name': name, 'slug': slug, 'tools': [],
                'model': model, 'prompt': prompt, 'description': description, 'temperature': temperature,
            }
            index_payload = {
                'id': agent_id, 'name': name, 'visibility': visibility,
                'owner_name': username, 'owner_email': owner_email,
                'description': description, 'score': r.get('score'),
            }
            try:
                agent_obj = AgentModel(**agent_payload) if AgentModel else agent_payload
            except Exception:
                agent_obj = agent_payload  # be permissive

            # Persist agent
            ok_a, res_a = _safe_call("save_agent", agent_obj)
            if not ok_a:
                errors += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'error',
                                   'msg': f"save_agent: {res_a}"})
                prog.progress(min(100, int(i*100/total))); continue

            # Try to recover canonical id from response
            try:
                if isinstance(res_a, dict):
                    agent_id = res_a.get('id') or res_a.get('agent_id') or agent_id
                elif hasattr(res_a, 'id'):
                    agent_id = getattr(res_a, 'id') or agent_id
            except Exception:
                pass
            index_payload['id'] = agent_id

            # Persist index
            ok_i, res_i = _safe_call("save_index", index_payload)
            if not ok_i:
                partial += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'partial',
                                   'msg': f"save_index: {res_i}", 'agent_id': agent_id})
                prog.progress(min(100, int(i*100/total))); continue

            # Optional share (wrap; we’re not editing s3_agent_manager.py)
            if shared_with:
                _safe_call("share_agent", agent_id, shared_with)

            created += 1
            resultados.append({'name': name, 'owner': owner_email, 'status': 'ok',
                               'msg': '', 'agent_id': agent_id})
            prog.progress(min(100, int(i*100/total)))

        st.success(f'Concluído: criados={created}, pulados={skipped}, parciais={partial}, erros={errors}')
        st.dataframe(resultados)
# =================== /Criar Agentes (JSON) — FIXED BRANCH ======================


            prog.progress(min(100, int(i*100/len(data))))

        st.success(f'Concluído: criados={created}, pulados={skipped}, erros={errors}')
        st.dataframe(resultados)
