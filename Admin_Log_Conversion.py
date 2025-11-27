#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# IMPORTS DO PROJETO
# -------------------------------------------------------------------
# Ajuste estes imports conforme sua base de código real.
# Aqui assumimos:
#   - eva_logger: objeto que sabe baixar logs (download_logs)
#   - eva_dashboard: função que mostra o dashboard
#   - etc.
# Se estes já são importados em outro lugar no seu Admin.py original,
# mantenha os imports de lá e remova/ajuste estes.
# -------------------------------------------------------------------
# from ema.core.eva_logger import eva_logger
# from ema.core.dashboard import eva_dashboard
# ...

# Importa helpers centralizados
from Helper_Admin import (
    compute_scores_from_df,
    parse_log_text_to_dataframe,  # se você não tiver, pode não usar
)

# -------------------------------------------------------------------
# Helpers para conversão de logs (EMA → formato legado)
# -------------------------------------------------------------------

def _find_column_by_substring(df: pd.DataFrame, substrings) -> str | None:
    """
    Procura a primeira coluna cujo nome contenha algum dos substrings indicados.
    Retorna o nome da coluna ou None se nada for encontrado.
    """
    substrings = [s.lower() for s in substrings]
    for col in df.columns:
        name = str(col).lower()
        if any(sub in name for sub in substrings):
            return col
    return None


def convert_logs_text_to_normalized_df(logs_text: str, debug: bool = False) -> pd.DataFrame:
    """
    Recebe o texto bruto de logs (.log com JSON lines, possivelmente no formato EMA)
    e retorna um DataFrame normalizado no formato legado, com colunas:

        timestamp, user, page, message, __line__, type, selected_agent, model

    Lógica:
      - parseia JSON lines → df_raw;
      - se df_raw já tiver todas as colunas esperadas → usa diretamente;
      - caso contrário, tenta inferir colunas de user/model/agent etc. via heurística.
    """

    # 1) Parse genérico: JSON por linha
    rows = []
    for i, line in enumerate(logs_text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            # Linha não é JSON válido → ignora
            continue
        if not isinstance(obj, dict):
            obj = {"value": obj}
        obj.setdefault("__line__", i)
        rows.append(obj)

    if not rows:
        if debug:
            st.write("⚠️ Nenhuma linha JSON válida encontrada no log.")
        cols = ["timestamp", "user", "page", "message", "__line__", "type", "selected_agent", "model"]
        return pd.DataFrame(columns=cols)

    df_raw = pd.DataFrame(rows)

    expected = ["timestamp", "user", "page", "message", "__line__", "type", "selected_agent", "model"]

    # 2) Caso trivial: já está no formato legado
    if set(expected).issubset(df_raw.columns):
        df = df_raw.copy()
    else:
        # 3) Formato "novo"/EMA → heurística de colunas
        if debug:
            st.write("♻️ Convertendo logs EMA para formato legado (heurística de colunas)...")
            st.write("Colunas disponíveis em df_raw:", list(df_raw.columns))

        df = df_raw.copy()

        ts_col    = _find_column_by_substring(df, ["timestamp", "time", "created_at"])
        user_col  = _find_column_by_substring(df, ["user", "email", "actor"])
        page_col  = _find_column_by_substring(df, ["page", "screen", "context"])
        msg_col   = _find_column_by_substring(df, ["message", "content", "text"])
        type_col  = _find_column_by_substring(df, ["type", "event_type", "kind"])
        agent_col = _find_column_by_substring(df, ["selected_agent", "agent", "assistant", "bot"])
        model_col = _find_column_by_substring(df, ["model", "llm_model"])

        if debug:
            st.write("Mapeamento de colunas inferido:")
            st.write(f"  timestamp      ← {ts_col}")
            st.write(f"  user           ← {user_col}")
            st.write(f"  page           ← {page_col}")
            st.write(f"  message        ← {msg_col}")
            st.write(f"  type           ← {type_col}")
            st.write(f"  selected_agent ← {agent_col}")
            st.write(f"  model          ← {model_col}")

        df_norm = pd.DataFrame()

        df_norm["timestamp"]      = df[ts_col]    if ts_col    else np.nan
        df_norm["user"]           = df[user_col]  if user_col  else np.nan
        df_norm["page"]           = df[page_col]  if page_col  else "chat"
        df_norm["message"]        = df[msg_col]   if msg_col   else ""
        df_norm["__line__"]       = df["__line__"] if "__line__" in df.columns else pd.Series(range(1, len(df) + 1))
        df_norm["type"]           = df[type_col]  if type_col  else np.nan
        df_norm["selected_agent"] = df[agent_col] if agent_col else np.nan
        df_norm["model"]          = df[model_col] if model_col else np.nan

        df = df_norm

    # 4) Garante todas as colunas esperadas
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # 5) Normalizações de tipo / formatação
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["user"] = df["user"].astype(str).str.strip().str.lower()
    df["page"] = df["page"].astype(str).str.strip().str.lower()
    df["selected_agent"] = df["selected_agent"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    return df


# -------------------------------------------------------------------
# UI: datas padrão (exemplo)
# -------------------------------------------------------------------

def _default_dates():
    hoje = date.today()
    return hoje, hoje




    
elif opcao == "Atualizar Scores (Logs)":
    st.subheader("Atualizar scores de agentes a partir dos logs de acesso")

    col1, col2, col3 = st.columns(3)
    with col1:
        alpha = st.number_input(
            "Peso de diversidade (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
        )
    with col2:
        dry_run = st.checkbox("Dry run (não salvar no index.yaml)", value=True)
    with col3:
        debug = st.checkbox("Debug", value=False)

    origem_logs = st.radio(
        "Origem dos logs",
        options=["Baixar logs do EVA", "Enviar arquivo .log local"],
        index=0,
    )

    uploaded_file = None
    if origem_logs == "Enviar arquivo .log local":
        uploaded_file = st.file_uploader(
            "Envie um arquivo .log (JSON lines)", type=["log", "txt"]
        )

    st.markdown("---")

    if st.button("Executar atualização de scores", type="primary"):
        # -------------------- 1) Obter texto dos logs --------------------
        if origem_logs == "Baixar logs do EVA":
            st.info("📥 Baixando logs a partir do EVA Logger…")
            # Aqui assumimos que eva_logger já existe (como no Admin original)
            content = eva_logger.download_logs(
                start_date=start_date, end_date=end_date
            )
            if isinstance(content, bytes):
                logs_text = content.decode("utf-8", errors="replace")
            else:
                logs_text = str(content)
        else:
            if uploaded_file is None:
                st.error("Envie um arquivo de log antes de executar.")
                st.stop()
            st.info("📥 Lendo arquivo de log enviado…")
            logs_text = uploaded_file.read().decode("utf-8", errors="replace")

        # -------------------- 2) Converter → DataFrame legado --------------------
        st.info("🔄 Convertendo e normalizando logs para formato legado…")
        df_raw = convert_logs_text_to_normalized_df(logs_text, debug=debug)

        if debug:
            st.markdown("#### DataFrame bruto normalizado")
            st.dataframe(df_raw.head())
            st.write("value_counts(page):")
            st.write(df_raw["page"].value_counts(dropna=False))

        # -------------------- 3) Filtros de política --------------------
        st.info("🧹 Aplicando filtros de política…")
        mask_valid = (
            (df_raw["page"] != "landing")
            & df_raw["selected_agent"].notna()
            & (df_raw["selected_agent"] != "")
            & df_raw["model"].notna()
            & (df_raw["model"] != "")
        )
        df_filt = df_raw.loc[mask_valid].copy()

        if debug:
            st.markdown("#### DataFrame após filtros")
            st.dataframe(df_filt.head())

        if df_filt.empty:
            st.warning("Após os filtros, nenhum log restante. Nada a atualizar.")
            st.stop()

        # -------------------- 4) Calcular scores --------------------
        st.info("📊 Calculando scores por agente…")
        df_scores = compute_scores_from_df(df_filt, alpha=alpha)

        # remove agentes vazios
        df_scores = df_scores[
            df_scores["selected_agent"].notna()
            & (df_scores["selected_agent"].astype(str).str.strip() != "")
        ].copy()

        score_map = {
            str(row["selected_agent"]).strip().lower(): float(row["score"])
            for _, row in df_scores.iterrows()
        }

        if debug:
            st.markdown("#### Scores calculados")
            st.dataframe(df_scores)

        # -------------------- 5) Obter o manager --------------------
        mgr = (
            st.session_state.get("multi_agent_manager")
            or st.session_state.get("agent_manager")
        )

        if mgr is None:
            st.warning(
                "Nenhum manager encontrado em st.session_state "
                "(chaves tentadas: 'multi_agent_manager', 'agent_manager').\n"
                "Os scores foram calculados, mas o index.yaml não será atualizado."
            )
            st.stop()

        has_index_api = hasattr(mgr, "load_index") and hasattr(mgr, "edit_index")

        updated = 0
        preview_rows: list[dict] = []

        # -------------------- 6) Função de patch com penalização --------------------
        def patch_index(index_list):
            """
            - Se o agente aparece nos logs → score novo de score_map.
            - Se NÃO aparece → penaliza score antigo dividindo por 10.
            """
            nonlocal updated, preview_rows

            for item in index_list:
                if not isinstance(item, dict):
                    continue

                name_raw = item.get("name") or ""
                name_norm = name_raw.strip().lower()
                old_score = item.get("score", 0.0)

                try:
                    old_score_val = float(old_score or 0.0)
                except Exception:
                    old_score_val = 0.0

                if name_norm in score_map:
                    new_score = float(score_map[name_norm])
                    status = "atualizado_com_logs"
                else:
                    if old_score is None:
                        new_score = 0.0
                    else:
                        new_score = old_score_val / 10.0
                    status = "penalizado_sem_logs"

                preview_rows.append(
                    {
                        "name": name_raw,
                        "old_score": old_score_val,
                        "new_score": new_score,
                        "status": status,
                    }
                )

                if not dry_run:
                    item["score"] = new_score

                if status in ("atualizado_com_logs", "penalizado_sem_logs"):
                    updated += 1

            return index_list

        # -------------------- 7) Aplicar / simular atualização do index --------------------
        if not has_index_api:
            st.warning(
                "O manager atual não expõe a API de índice (load_index/edit_index).\n"
                "Os scores foram calculados, mas o index.yaml não será atualizado automaticamente."
            )
        else:
            if dry_run:
                st.info(
                    "🔍 Dry run: simulando atualização de scores (nada será salvo)."
                )
                try:
                    index_current = mgr.load_index()
                    if index_current is None:
                        st.error("mgr.load_index() retornou None.")
                        st.stop()
                    _ = patch_index(index_current)
                    st.success(
                        f"Dry run concluído. {updated} agentes seriam atualizados "
                        "(incluindo penalizações de agentes sem logs)."
                    )
                except Exception as e:
                    st.error(f"Erro ao carregar/patch index em dry run: {e}")
                    st.stop()
            else:
                st.info(
                    "💾 Atualizando index.yaml no backend (via MultiAgentManager)…"
                )
                try:
                    mgr.edit_index(patch_index)
                    st.success(
                        f"Scores atualizados/penalizados para {updated} agentes."
                    )
                except Exception as e:
                    st.error(f"Erro ao aplicar edit_index: {e}")
                    st.stop()

        # -------------------- 8) Preview das alterações --------------------
        if preview_rows:
            df_preview = pd.DataFrame(preview_rows)
            st.markdown("#### Resumo das alterações de score")
            st.dataframe(df_preview)
        else:
            st.info(
                "Nenhum agente do index.yaml foi encontrado para atualização/penalização."
            )
