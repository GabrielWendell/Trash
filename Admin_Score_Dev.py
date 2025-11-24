# ============================================================
#  OPTION: Atualizar Scores (Logs)
# ============================================================

elif opcao == "Atualizar Scores (Logs)":

    st.subheader("Atualizar scores de agentes a partir dos logs de acesso")

    # ----------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------
    alpha = st.number_input("Peso de diversidade (alpha)", 0.0, 1.0, 0.15, 0.01)
    dry_run = st.checkbox("Dry run (não salvar no index.yaml)", value=True)

    st.markdown("### Escolha a origem dos logs")
    log_mode = st.radio(
        "Selecione:",
        ["Baixar logs do EVA", "Enviar arquivo .log local"]
    )

    # ======================================================================
    #  Helper Functions
    # ======================================================================

    REQUIRED_COLS = [
        "timestamp", "user", "page", "message",
        "__line__", "type", "selected_agent", "model"
    ]

    def df_from_log_text(text: str) -> pd.DataFrame:
        """
        Convert EVA raw log text (JSON lines) into a DataFrame.
        """
        import json

        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                continue
        return pd.DataFrame(rows)

    # ----------------------------------------------------------

    def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        for c in REQUIRED_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[REQUIRED_COLS]

    # ----------------------------------------------------------

    def standardize(df):
        df["user"] = df["user"].astype(str).str.strip().str.lower()
        df["selected_agent"] = df["selected_agent"].astype(str).str.strip()
        df["page"] = df["page"].astype(str).str.strip().str.lower()
        df["model"] = df["model"].astype(str).str.strip()
        return df

    # ----------------------------------------------------------

    def filter_policy(df):
        m1 = df["page"] != "landing"
        m2 = df["selected_agent"].notna() & (df["selected_agent"].str.len() > 0)
        m3 = df["model"].notna() & (df["model"].str.len() > 0)
        return df[m1 & m2 & m3]

    # ----------------------------------------------------------

    def log_normalize(series: pd.Series) -> pd.Series:
        maxv = float(series.max()) if len(series) else 0
        if maxv <= 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return np.log1p(series) / np.log1p(maxv)

    # ----------------------------------------------------------

    def compute_scores(df, alpha):
        """
        Returns:
            dict[name_lower → score]
        """
        groups = df.groupby("selected_agent")

        score_map = {}

        for ag, g in groups:
            w_sum = g.shape[0]
            u_users = g["user"].nunique()

            # Diversity
            per_user = g.groupby("user").size()
            tot = per_user.sum()
            if tot <= 0:
                diversity = 0
            else:
                shares = per_user / tot
                hhi = (shares**2).sum()
                diversity = max(0, min(1, 1 - hhi))

            # ScoreH
            na = log_normalize(pd.Series({"x": w_sum})).iloc[0]
            nu = log_normalize(pd.Series({"x": u_users})).iloc[0]

            denom = na + nu
            if denom <= 0:
                scoreH = 0
            else:
                scoreH = 2 * na * nu / denom

            score = scoreH * (alpha + (1 - alpha) * diversity)

            score_map[ag.strip().lower()] = float(score)

        return score_map

    # ======================================================================
    #  Read Logs
    # ======================================================================

    if log_mode == "Baixar logs do EVA":
        st.info("Baixando logs a partir do EVA Logger…")
        logs_text = eva_logger.download_logs(start_date=start_date, end_date=end_date)

    else:  # enviar arquivo local (.log)
        uploaded = st.file_uploader("Envie um arquivo .log", type=["log"])
        if uploaded is None:
            st.stop()

        logs_text = uploaded.read().decode("utf-8", errors="ignore")

    # ======================================================================
    # Convert logs → DataFrame
    # ======================================================================
    st.info("Convertendo logs para DataFrame…")
    df_raw = df_from_log_text(logs_text)
    df_raw = ensure_columns(df_raw)
    df_raw = standardize(df_raw)

    # ======================================================================
    # Filter
    # ======================================================================
    st.info("Aplicando filtros…")
    df_filt = filter_policy(df_raw)

    if df_filt.empty:
        st.warning("Após filtros, nenhum log restante.")
        st.stop()

    # ======================================================================
    # Compute Scores
    # ======================================================================
    st.info("Calculando scores…")
    score_map = compute_scores(df_filt, alpha)

    # ======================================================================
    #  Load MultiAgentManager
    # ======================================================================
    from ema.core.s3_agent_manager import MultiAgentManager
    mgr = MultiAgentManager()

    # ======================================================================
    # Patch Function
    # ======================================================================
    updated = 0

    def patch_index(index_list):
        nonlocal updated
        for item in index_list:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip().lower()
            if not name:
                continue
            if name in score_map:
                item["score"] = float(score_map[name])
                updated += 1
        return index_list

    # ======================================================================
    # Apply Update
    # ======================================================================
    if dry_run:
        st.info("Dry run: preview das alterações (não serão salvas).")
        index_current = mgr.load_index()
        patched = patch_index(index_current)

        st.success(f"Dry run concluído. {updated} agentes seriam atualizados.")
        st.dataframe(pd.DataFrame(patched))
    else:
        st.info("Atualizando index.yaml…")
        mgr.edit_index(patch_index)
        st.success(f"Scores atualizados para {updated} agentes no index.yaml!")
