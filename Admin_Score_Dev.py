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

        # 1) Obter texto de logs
        if origem_logs == "Baixar logs do EVA":
            st.info("📥 Baixando logs a partir do EVA Logger…")
            content = eva_logger.download_logs(
                start_date=start_date, end_date=end_date
            )
            # content pode ser bytes ou str dependendo do logger
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

        # 2) Converter texto → DataFrame bruto
        st.info("🔄 Convertendo logs para DataFrame…")
        df_raw = parse_log_text_to_dataframe(logs_text)

        # Garante colunas mínimas
        needed_cols = [
            "timestamp",
            "user",
            "page",
            "__line__",
            "type",
            "selected_agent",
            "model",
        ]
        for c in needed_cols:
            if c not in df_raw.columns:
                df_raw[c] = np.nan
        df_raw = df_raw[needed_cols]

        # Limpeza básica
        df_raw["user"] = df_raw["user"].astype(str).str.strip().str.lower()
        df_raw["page"] = df_raw["page"].astype(str).str.strip().str.lower()
        df_raw["selected_agent"] = df_raw["selected_agent"].astype(str).str.strip()
        df_raw["model"] = df_raw["model"].astype(str).str.strip()

        if debug:
            st.markdown("#### DataFrame bruto")
            st.write(df_raw.head())
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
            st.write(df_filt.head())

        if df_filt.empty:
            st.warning("Após os filtros, nenhum log restante. Nada a atualizar.")
            st.stop()

        # 4) Calcular scores
        st.info("📊 Calculando scores por agente…")
        df_scores = compute_scores_from_df(df_filt, alpha=alpha)

        # Remove linhas cujo selected_agent é NaN ou string vazia
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

        # 5) Obter o manager já usado no app
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

        # 6) Função de patch que será aplicada ao index.yaml
        updated = 0
        preview_rows = []

        def patch_index(index_list):
            """
            Atualiza / penaliza scores no index.yaml.

            - Se o agente aparece nos logs → usa score de score_map.
            - Se NÃO aparece nos logs → penaliza score antigo dividindo por 10.
            """
            nonlocal updated, preview_rows

            for item in index_list:
                if not isinstance(item, dict):
                    continue

                name_raw = item.get("name") or ""
                name_norm = name_raw.strip().lower()
                old_score = item.get("score", 0.0)

                # Normaliza old_score para float
                try:
                    old_score_val = float(old_score or 0.0)
                except Exception:
                    old_score_val = 0.0

                if name_norm in score_map:
                    # Agente encontrado nos logs → usa novo score
                    new_score = float(score_map[name_norm])
                    status = "atualizado_com_logs"
                else:
                    # Agente NÃO apareceu nos logs → penaliza score antigo
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

                # Conta qualquer agente que sofreu mudança (atualização ou penalização)
                if status in ("atualizado_com_logs", "penalizado_sem_logs"):
                    updated += 1

            return index_list

        # 7) Aplicar / simular atualização do index
        if not has_index_api:
            st.warning(
                "O objeto de manager atual não expõe a API de índice "
                "(load_index/edit_index). Os scores foram calculados, "
                "mas o index.yaml não será atualizado automaticamente."
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

        # 8) Mostrar tabela de preview/relatório
        if preview_rows:
            df_preview = pd.DataFrame(preview_rows)
            st.markdown("#### Resumo das alterações de score")
            st.dataframe(df_preview)
        else:
            st.info(
                "Nenhum agente do index.yaml foi encontrado para atualização/penalização."
            )
