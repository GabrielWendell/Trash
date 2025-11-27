from Helper_Admin import parse_log_text_to_dataframe, compute_scores_from_df
from log_conversion import convert_logs as convert_ema_records_to_legacy_records
import json
import numpy as np
import pandas as pd

# ------------------------------------------------------
# Detecção e conversão de formato de logs (EMA → legado)
# ------------------------------------------------------

def detect_log_format(record: dict) -> str:
    """
    Detecta se um registro de log já está no formato legado
    ou se parece ser um log novo do EMA.
    Ajuste as condições de acordo com a estrutura real.
    """
    if not isinstance(record, dict):
        return "unknown"

    keys = set(record.keys())

    # Formato legado clássico: já tem page, selected_agent, model
    if {"page", "selected_agent", "model"}.issubset(keys):
        return "legacy"

    # Heurística para EMA: ajuste essas chaves aos seus logs reais
    if "ema_version" in keys or "agents" in keys or "multiagent" in keys:
        return "ema"

    # Fallback: tratar como legado para não converter o que já funciona
    return "legacy"


def convert_logs_text_to_normalized_df(logs_text: str, debug: bool = False) -> pd.DataFrame:
    """
    Recebe o texto bruto do log (.log com JSON lines, possivelmente EMA),
    detecta o formato e retorna um DataFrame no formato *legado*,
    com colunas padrão:
        timestamp, user, page, message, __line__, type, selected_agent, model
    """

    # 1) Parse básico em JSON lines
    records = []
    for i, line in enumerate(logs_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            if debug:
                st.write(f"⚠️ Linha {i} não é JSON válido, ignorando.")
            continue
        if not isinstance(obj, dict):
            obj = {"value": obj}
        # manter número da linha, se já não existir
        obj.setdefault("__line__", i)
        records.append(obj)

    if not records:
        if debug:
            st.write("⚠️ Nenhum registro JSON válido encontrado no log.")
        cols = ["timestamp", "user", "page", "message", "__line__", "type", "selected_agent", "model"]
        return pd.DataFrame(columns=cols)

    # 2) Detecta formato com base no primeiro registro
    fmt = detect_log_format(records[0])
    if debug:
        st.write(f"🔍 Formato detectado: {fmt}")

    legacy_records = []

    if fmt == "legacy":
        # 3a) Já está no formato antigo → só normaliza as chaves
        for rec in records:
            legacy_records.append(
                {
                    "timestamp":      rec.get("timestamp"),
                    "user":           rec.get("user"),
                    "page":           rec.get("page"),
                    "message":        rec.get("message", ""),
                    "__line__":       rec.get("__line__"),
                    "type":           rec.get("type"),
                    "selected_agent": rec.get("selected_agent"),
                    "model":          rec.get("model"),
                }
            )
    else:
        # 3b) É EMA → delega conversão ao log_conversion.py
        if debug:
            st.write("♻️ Convertendo logs EMA para formato legado usando log_conversion.py…")

        # Aqui supomos que convert_ema_records_to_legacy_records recebe uma lista de dicts
        # e devolve uma lista de dicts já no formato legado.
        ema_legacy = convert_ema_records_to_legacy_records(records)

        for rec in ema_legacy:
            legacy_records.append(
                {
                    "timestamp":      rec.get("timestamp"),
                    "user":           rec.get("user"),
                    "page":           rec.get("page"),
                    "message":        rec.get("message", ""),
                    "__line__":       rec.get("__line__", None),
                    "type":           rec.get("type"),
                    "selected_agent": rec.get("selected_agent"),
                    "model":          rec.get("model"),
                }
            )

    # 4) Monta DataFrame final com colunas padrão
    df = pd.DataFrame(legacy_records)

    expected_cols = ["timestamp", "user", "page", "message", "__line__", "type", "selected_agent", "model"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Conversões de tipo básicas
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["user"] = df["user"].astype(str).str.strip().str.lower()
    df["page"] = df["page"].astype(str).str.strip().str.lower()
    df["selected_agent"] = df["selected_agent"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()

    if debug:
        st.write("✅ DataFrame normalizado (formato legado):")
        st.dataframe(df.head())

    return df


