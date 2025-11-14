elif opcao == 'Atualizar Scores (Logs)':
    import json
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import yaml

    st.subheader("Atualizar scores de agentes a partir dos logs de acesso")

    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
    with col_a:
        alpha = st.number_input("Peso de diversidade (alpha)", 0.0, 1.0, 0.15, 0.01)
    with col_b:
        dry_run_scores = st.checkbox("Dry run (não salvar no index.yaml)", value=True)
    with col_c:
        debug_logs = st.checkbox("Mostrar debug dos logs", value=True)
    with col_d:
        use_local_logs = st.checkbox("Usar log local (CSV)", value=False)

    # caminho padrão do CSV local (../../../../Backup/data/logs_csv/...)
    local_csv_path = (
        Path(__file__).resolve().parents[4]
        / "Backup"
        / "data"
        / "logs_csv"
        / "logs-2025-11-14-15-00-55.csv"
    )
    if use_local_logs:
        st.caption(f"Log local: {local_csv_path}")

    # ------------------ Helpers internos ------------------ #

    def _df_from_log_text(text: str) -> pd.DataFrame:
        """Converte o conteúdo do .log (JSONL) em DataFrame (similar ao parse_logs)."""
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                obj = {"value": obj}
            obj["__line__"] = i
            rows.append(obj)
        return pd.DataFrame(rows)

    def _ensure_cols(df: pd.DataFrame, cols):
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        df["user"] = df["user"].astype(str).str.strip().str.lower()
        df["page"] = df["page"].astype(str).str.strip().str.lower()
        df["selected_agent"] = df["selected_agent"].astype(str).str.strip()
        df["model"] = df["model"].astype(str).str.strip()
        return df

    def _empty_like(s: pd.Series) -> pd.Series:
        s_str = s.astype(str).str.strip().str.lower()
        return s.isna() | s_str.isin(["", "nan", "none", "null"])

    def _policy_filter(df: pd.DataFrame):
        stats = {}
        n0 = len(df)

        mask = df["page"] == "landing"
        stats["dropped_landing"] = int(mask.sum())
        df = df.loc[~mask]

        mask = _empty_like(df["selected_agent"])
        stats["dropped_selected_agent_invalid"] = int(mask.sum())
        df = df.loc[~mask]

        mask = _empty_like(df["model"])
        stats["dropped_model_invalid"] = int(mask.sum())
        df = df.loc[~mask]

        stats["kept_after_filters"] = int(len(df))
        stats["dropped_total"] = int(n0 - len(df))
        return df, stats

    def _log_normalize(s: pd.Series) -> pd.Series:
        maxv = float(s.max()) if len(s) else 0.0
        if maxv <= 0.0:
            return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
        return np.log1p(s.astype(float)) / math.log1p(maxv)

    # ------------------ Ação do botão ------------------ #

    if st.button("Baixar logs, calcular scores e atualizar index.yaml", type="primary"):
        # 1) Obter logs
        if use_local_logs:
            st.info(f"Lendo log local CSV: {local_csv_path}")
            if not local_csv_path.exists():
                st.error(f"Arquivo CSV local não encontrado em: {local_csv_path}")
                st.stop()
            # CSV já convertido (parse_logs.py); lê direto
            df = pd.read_csv(local_csv_path, engine="python", dtype=str)
        else:
            st.info("Baixando logs a partir do EVA Logger…")
            content = eva_logger.download_logs(start_date=start_date, end_date=end_date)
            if not content:
                st.error("Nenhum log retornado para o intervalo selecionado.")
                st.stop()
            df = _df_from_log_text(content)

        # 2) Garantir colunas e padronizar
        st.info("Convertendo logs para DataFrame…")
        req_cols = ["timestamp", "user", "page", "message", "__line__", "type", "selected_agent", "model"]
        df = _ensure_cols(df, req_cols)
        df = _standardize(df)

        if debug_logs:
            st.markdown("### Debug — DataFrame bruto")
            st.write(f"Linhas totais (bruto): {len(df)}")
            st.write("Colunas:", list(df.columns))
            st.write("Amostra (head):")
            st.dataframe(df.head(50))
            st.write("value_counts(page):")
            st.write(df["page"].value_counts(dropna=False))
            st.write("value_counts(selected_agent):")
            st.write(df["selected_agent"].value_counts(dropna=False).head(20))
            st.write("value_counts(model):")
            st.write(df["model"].value_counts(dropna=False))

        # 3) Filtros
        st.info("Aplicando filtros (page!='landing', selected_agent/model válidos)…")
        df, filt_stats = _policy_filter(df)

        if debug_logs:
            st.markdown("### Debug — Após filtros")
            st.write("Stats dos filtros:", filt_stats)
            st.write(f"Linhas restantes: {len(df)}")
            st.dataframe(df.head(50))

        if df.empty:
            st.warning("Após os filtros, nenhum log restante. Nada a atualizar.")
            st.stop()

        # 4) Métricas e scores
        st.info("Calculando métricas e scores por agente…")

        records = []
        grouped = df.groupby("selected_agent", dropna=True)

        for ag_name, g in grouped:
            messages = float(len(g))
            unique_users = int(g["user"].nunique())
            per_user = g["user"].value_counts(normalize=True)
            hhi = float((per_user ** 2).sum()) if len(per_user) else 1.0
            diversity = max(0.0, min(1.0, 1.0 - hhi))
            records.append(
                {
                    "name": ag_name,
                    "messages": messages,
                    "unique_users": unique_users,
                    "hhi": hhi,
                    "diversity": diversity,
                }
            )

        if not records:
            st.warning("Nenhum agente encontrado nos logs filtrados.")
            st.stop()

        df_metrics = pd.DataFrame(records)

        na = _log_normalize(df_metrics["messages"])
        nu = _log_normalize(df_metrics["unique_users"])
        scoreH = (2.0 * na * nu) / (na + nu)
        scoreH[(na + nu) == 0.0] = 0.0

        div = df_metrics["diversity"].astype(float).clip(0, 1)
        score = scoreH * (alpha + (1.0 - alpha) * div)

        df_metrics["scoreH"] = scoreH
        df_metrics["score"] = score

        score_map = {
            str(row["name"]).strip().lower(): float(row["score"])
            for _, row in df_metrics.iterrows()
        }

        st.success(f"Scores calculados para {len(score_map)} agentes distintos.")

        # 5) Atualizar index.yaml (mesmo que antes)
        index_path = Path(__file__).resolve().parents[3] / "mock_bucket" / "ema" / "index" / "index.yaml"
        if not index_path.exists():
            st.error(f"Arquivo index.yaml não encontrado em: {index_path}")
            st.stop()

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = yaml.safe_load(f) or []

        if not isinstance(index_data, list):
            st.error("Conteúdo de index.yaml não é uma lista de agentes.")
            st.stop()

        rows_view = []
        updated = 0

        for item in index_data:
            if not isinstance(item, dict):
                continue
            name_idx = (item.get("name") or "").strip()
            key = name_idx.lower()
            old_score = float(item.get("score") or 0.0)
            new_score = old_score
            status = "sem_logs"

            if key in score_map:
                new_score = float(score_map[key])
                status = "atualizado"
                if not dry_run_scores:
                    item["score"] = new_score
                updated += 1

            rows_view.append(
                {
                    "name": name_idx,
                    "old_score": old_score,
                    "new_score": new_score,
                    "status": status,
                }
            )

        if not dry_run_scores:
            with open(index_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(index_data, f, allow_unicode=True, sort_keys=False)
            st.success(f"index.yaml salvo. {updated} agentes tiveram score atualizado.")
        else:
            st.info(f"Dry run ativado: {updated} agentes seriam atualizados (index.yaml não foi modificado).")

        st.dataframe(pd.DataFrame(rows_view))
