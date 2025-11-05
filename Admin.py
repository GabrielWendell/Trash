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
elif opcao == 'Criar Agentes (JSON)':
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

        # Resolve gerenciador já autenticado (definido em outro ponto do app)
        try:
            mgr = st.session_state['multi_agent_manager']  # MultiAgentManager
        except KeyError:
            st.error('multi_agent_manager não encontrado em st.session_state.')
            st.stop()

        created = 0; skipped = 0; errors = 0
        resultados = []
        prog = st.progress(0)

        for i, r in enumerate(data, start=1):
            # Campos mínimos esperados no JSON filtrado
            name = (r.get('name') or '').strip()
            owner_email = (r.get('owner') or '').strip().lower()
            prompt = (r.get('prompt') or '').strip()
            model = (r.get('model') or '').strip()
            if not (name and owner_email and prompt and model):
                errors += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'error', 'msg': 'campos obrigatórios ausentes'})
                prog.progress(min(100, int(i*100/len(data))))
                continue

            agent_id = (r.get('id') or '').strip() or str(uuid.uuid4())
            description = (r.get('description') or '').strip()
            temperature = _clamp(r.get('temperature'), 0.0, 1.0, 0.0)
            visibility = r.get('visibility') or 'Privado'  # PT-BR já no master
            shared_with = r.get('shared_with') or []
            if isinstance(shared_with, dict):
                emails = set()
                for lst in shared_with.values():
                    if isinstance(lst, list):
                        emails.update([str(x).strip().lower() for x in lst if str(x).strip()])
                shared_with = sorted(emails)

            # Idempotência simples: verifica por (owner_email, slug)
            slug = slugify(name)
            try:
                indexes = mgr.list_indexes()  # deve retornar lista/dict
                items = indexes if isinstance(indexes, list) else indexes.get('items', [])
                if any((it.get('owner_email','').lower()==owner_email and slugify(it.get('name',''))==slug) for it in items):
                    skipped += 1
                    resultados.append({'name': name, 'owner': owner_email, 'status': 'skip', 'msg': 'já existe'})
                    prog.progress(min(100, int(i*100/len(data))))
                    continue
            except Exception:
                pass  # se lista falhar, tentamos criar assim mesmo

            # Monta payloads
            agent_def = {
                'id': agent_id,
                'name': name,
                'slug': slug,
                'tools': [],
                'model': model,
                'prompt': prompt,
                'description': description,
                'temperature': temperature,
            }
            index_def = {
                'id': agent_id,
                'name': name,
                'owner_name': r.get('username') or owner_email.split('@')[0],
                'owner_email': owner_email,
                'visibility': visibility,
                'description': description,
                'score': r.get('score'),
            }

            try:
                a = AgentModel(**agent_def)
            except Exception:
                a = agent_def
            try:
                i_def = IndexModel(**index_def)
            except Exception:
                i_def = index_def

            # Persistência
            try:
                mgr.save_agent(a)
                mgr.save_index(i_def)
                if shared_with:
                    try:
                        mgr.share_agent(agent_id, shared_with)
                    except Exception:
                        pass
                created += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'ok', 'msg': ''})
            except Exception as e:
                errors += 1
                resultados.append({'name': name, 'owner': owner_email, 'status': 'error', 'msg': str(e)})

            prog.progress(min(100, int(i*100/len(data))))

        st.success(f'Concluído: criados={created}, pulados={skipped}, erros={errors}')
        st.dataframe(resultados)
