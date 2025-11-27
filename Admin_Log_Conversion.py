elif opcao == "Atualizar Scores (Logs)":
    st.subheader("Atualizar scores de agentes a partir dos logs de acesso")

    # ---------------- UI de configuração ----------------
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

    # ---------------- Botão para executar ----------------
    if st.button("Executar atualização de scores", type="primary"):

        # 1) Obter texto dos logs
        if origem_logs == "Baixar logs do EVA":
            st.info("📥 Baixando logs a partir do EVA Logger…")
            content = eva_logger.download_logs(
                start_date=start_date,
                end_date=end_date,
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

        # 2) Converter texto → DataFrame legado normalizado
        st.info("🔄 Convertendo e normalizando logs para formato legado…")

        # --- Trata retorno como df OU (df, meta) ---
        res = convert_logs_text_to_normalized_df(logs_text, debug=debug)
        meta = {}

        if isinstance(res, tuple):
            # Helper retornou (df, meta_dict)
            df_raw, meta = res
            if not isinstance(meta, dict):
                meta = {}
        else:
            # Helper retornou apenas o DataFrame
            df_raw = res
            if hasattr(df_raw, "attrs"):
                meta = df_raw.attrs
        # Campos de formato / conversão, com defaults seguros
        log_fmt = meta.get("log_format", "unknown")
        conv = bool(meta.get("conversion_applied", False))

        # ---------- Seção de debug sobre formato dos logs ----------
        if debug:
            st.markdown("### 📄 Resumo do formato de log utilizado")

            if log_fmt == "legacy":
                st.success("Formato detectado: **legado (antigo EVA)**.")
            elif log_fmt == "ema":
                st.warning("Formato detectado: **novo/EMA**.")
            else:
                st.info(
                    f"Formato detectado: `{log_fmt}` "
                    "(não identificado explicitamente)."
                )

            if conv:
                st.info(
                    "Uma **conversão EMA → formato legado** foi aplicada antes da análise."
                )
            else:
                st.info(
                    "Nenhuma conversão de formato foi necessária "
                    "(logs já estavam no padrão legado)."
                )

            # Evita erro de timezone/offset no st.dataframe:
            df_dbg = df_raw.copy()
            if "timestamp" in df_dbg.columns:
                # 👇 alteração: usar map(str) em vez de astype(str)
                df_dbg["timestamp"] = df_dbg["timestamp"].map(str)

            st.markdown("#### DataFrame bruto normalizado (formato legado)")
            st.dataframe(df_dbg.head())
            if "page" in df_raw.columns:
                st.write("value_counts(page):")
                st.write(df_raw["page"].value_counts(dropna=False))

        # 3) Filtros de política
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
            df_dbg2 = df_filt.copy()
            if "timestamp" in df_dbg2.columns:
                # 👇 mesma correção aqui
                df_dbg2["timestamp"] = df_dbg2["timestamp"].map(str)
            st.dataframe(df_dbg2.head())

        if df_filt.empty:
            st.warning("Após os filtros, nenhum log restante. Nada a atualizar.")
            st.stop()

        # 4) Calcular scores
        st.info("📊 Calculando scores por agente…")
        df_scores = compute_scores_from_df(df_filt, alpha=alpha)

        # Remove linhas cujo selected_agent é vazio ou NaN
        df_scores = df_scores[
            df_scores["selected_agent"].notna()
            & (df_scores["selected_agent"].astype(str).str.strip() != "")
        ].copy()

        # Mapa agent_name (lower) → score
        score_map = {
            str(row["selected_agent"]).strip().lower(): float(row["score"])
            for _, row in df_scores.iterrows()
        }

        if debug:
            st.markdown("#### Scores calculados")
            st.dataframe(df_scores)

        # 5) Obter manager
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
        if not has_index_api:
            st.warning(
                "O objeto manager não possui métodos 'load_index' e 'edit_index'.\n"
                "Os scores foram calculados, mas o index.yaml não será atualizado."
            )
            st.stop()

        # 6) Função de patch para aplicar nos dados do index.yaml
        updated = 0
        preview_rows = []

        def patch_index(index_list):
            nonlocal updated, preview_rows

            for item in index_list:
                if not isinstance(item, dict):
                    continue

                name_raw = item.get("name") or ""
                name_norm = name_raw.strip().lower()
                old_score = float(item.get("score", 0.0) or 0.0)

                if name_norm in score_map:
                    new_score = float(score_map[name_norm])
                    status = "updated_from_logs"
                else:
                    new_score = old_score / 10.0
                    status = "decayed_no_logs"

                if abs(new_score - old_score) < 1e-9:
                    continue

                item["score"] = new_score
                updated += 1
                preview_rows.append(
                    {
                        "name": name_raw,
                        "old_score": old_score,
                        "new_score": new_score,
                        "status": status,
                    }
                )

            return index_list

        # 7) Carregar index.yaml via manager
        try:
            st.info("🔎 Carregando index.yaml…")
            index_data = mgr.load_index()
            if not isinstance(index_data, list):
                index_data = []
        except Exception as e:
            st.error(f"Erro ao carregar index.yaml: {e}")
            st.stop()

        patched_index = patch_index(index_data)

        if debug and preview_rows:
            st.markdown("#### Pré-visualização das atualizações de score")
            st.dataframe(pd.DataFrame(preview_rows))

        # 8) Salvar ou simular (dry run)
        if dry_run:
            st.info("🔍 Dry run: simulando atualização de scores (nada será salvo).")
            st.success(f"{updated} agentes teriam seus scores atualizados.")
        else:
            try:
                st.info("💾 Salvando novos scores no index.yaml…")
                mgr.edit_index(patched_index)
                st.success(f"Scores atualizados para {updated} agentes.")
            except Exception as e:
                st.error(f"Erro ao salvar index.yaml: {e}")
