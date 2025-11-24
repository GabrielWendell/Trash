elif opcao == "Atualizar Scores (Logs)":
    st.subheader("Atualizar scores de agentes a partir dos logs de acesso")

    alpha = st.number_input("Peso da diversidade (alpha)", min_value=0.0, max_value=1.0,
                            value=0.15, step=0.01)

    dry_run = st.checkbox("Dry run (não salvar no index.yaml)", value=True)
    debug = st.checkbox("Debug")

    origem_logs = st.radio(
        "Escolha a origem dos logs",
        ("Baixar logs do EVA", "Enviar arquivo .log local")
    )

    uploaded_log = None
    if origem_logs == "Enviar arquivo .log local":
        uploaded_log = st.file_uploader("Envie um arquivo .log", type=["log"])

    st.markdown("---")

    # Only run when the user clicks the button
    if st.button("Executar atualização de scores"):
        try:
            st.info("Iniciando processo...")
            logs_text = ""

            # ---------------------------------------------------------
            # 1. Load logs
            # ---------------------------------------------------------
            if origem_logs == "Baixar logs do EVA":
                st.write("📥 Baixando logs a partir do EVA Logger...")
                content = eva_logger.download_logs(start_date, end_date)
                logs_text = content.decode("utf-8")

            else:
                if uploaded_log is None:
                    st.error("Você deve enviar um arquivo .log primeiro.")
                    st.stop()

                st.write("📥 Convertendo arquivo enviado...")
                logs_text = uploaded_log.read().decode("utf-8")

            # ---------------------------------------------------------
            # 2. Convert LOG → CSV DataFrame
            # ---------------------------------------------------------
            st.write("🔄 Convertendo logs para DataFrame...")
            df = parse_log_text_to_dataframe(logs_text)

            if "message" in df.columns:
                df = df.drop(columns=["message"])

            if debug:
                st.write("DataFrame original:")
                st.dataframe(df.head())

            # ---------------------------------------------------------
            # 3. Apply filters
            # ---------------------------------------------------------
            st.write("🔍 Aplicando filtros...")
            df_filt = df[
                (df["page"] != "landing") &
                (df["selected_agent"].notna()) &
                (df["model"].notna())
            ]

            if debug:
                st.write("Após filtros:")
                st.dataframe(df_filt.head())

            if df_filt.empty:
                st.warning("Após os filtros, nenhum log restante. Nada a atualizar.")
                st.stop()

            # ---------------------------------------------------------
            # 4. Compute scores
            # ---------------------------------------------------------
            st.write("📊 Calculando scores...")
            scores = compute_scores_from_df(df_filt, alpha)

            if debug:
                st.write("Scores calculados:")
                st.json(scores)

            # ---------------------------------------------------------
            # 5. Update index.yaml using MultiAgentManager
            # ---------------------------------------------------------
            user_email = st.session_state.get("user_email", "unknown@itau-unibanco.com.br")
            bucket = settings.S3_BUCKET  # adjust to match your project

            st.write("📁 Inicializando MultiAgentManager...")
            mgr = MultiAgentManager(bucket=bucket, user_email=user_email)

            updates_done = 0
            missing = 0

            for agent_name, sc in scores.items():
                agent_found = False

                index_list = mgr.load_index()
                if index_list is None:
                    st.error("Erro: index.yaml não pôde ser carregado.")
                    st.stop()

                # find agent in index.yaml
                for entry in index_list:
                    if entry["name"].strip().lower() == agent_name.strip().lower():
                        agent_found = True
                        if not dry_run:
                            mgr.edit_index(entry["id"], {"score": float(sc)})
                        updates_done += 1
                        break

                if not agent_found:
                    missing += 1

            # ---------------------------------------------------------
            # 6. Finished
            # ---------------------------------------------------------
            st.success(
                f"Concluído! Atualizados={updates_done}, "
                f"Não encontrados={missing}, Dry-run={dry_run}"
            )

        except Exception as e:
            st.error(f"Erro durante a execução: {e}")
            st.exception(e)
