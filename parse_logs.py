from pathlib import Path
import re
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd

# --------- Detectores simples ---------
APACHE_COMBINED_RE = re.compile(
    r'^(?P<host>\S+) (?P<ident>\S+) (?P<user>\S+) \[(?P<time>[^\]]+)\] '
    r'"(?P<request>[^"]*)" (?P<status>\d{3}) (?P<size>\S+) '
    r'"(?P<referer>[^"]*)" "(?P<agent>[^"]*)"'
)

APACHE_COMMON_RE = re.compile(
    r'^(?P<host>\S+) (?P<ident>\S+) (?P<user>\S+) \[(?P<time>[^\]]+)\] '
    r'"(?P<request>[^"]*)" (?P<status>\d{3}) (?P<size>\S+)'
)

KV_PAIR_RE = re.compile(r'([A-Za-z0-9_.\-]+)=(".*?"|\'.*?\'|\S+)')

# --------- Parsers ---------
def parse_json_lines(path: Path) -> Optional[pd.DataFrame]:
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    obj["line_number"] = i
                    rows.append(obj)
                else:
                    # json válido, mas não dict — tratar como mensagem
                    rows.append({"message": obj, "line_number": i})
            except json.JSONDecodeError:
                return None
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

def parse_kv_pairs(path: Path) -> Optional[pd.DataFrame]:
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            pairs = KV_PAIR_RE.findall(s)
            if not pairs:
                return None
            row = {k: v.strip('"\'') for k, v in pairs}
            row["line_number"] = i
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def parse_apache_like(path: Path) -> Optional[pd.DataFrame]:
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        matched_any = False
        for i, line in enumerate(f, 1):
            s = line.rstrip('\n')
            m = APACHE_COMBINED_RE.match(s) or APACHE_COMMON_RE.match(s)
            if not m:
                if i <= 10:
                    # nos primeiros sinais já falhou -> não é apache
                    return None
                else:
                    # pode haver linhas em branco/ruído, apenas ignora
                    continue
            matched_any = True
            row = m.groupdict()
            row["line_number"] = i
            rows.append(row)
    if not matched_any:
        return None
    df = pd.DataFrame(rows)
    # normaliza campos
    if "time" in df.columns:
        # exemplo: 10/Oct/2000:13:55:36 -0700
        df["timestamp"] = pd.to_datetime(df["time"], format="%d/%b/%Y:%H:%M:%S %z", errors="coerce")
    if "status" in df.columns:
        df["status"] = pd.to_numeric(df["status"], errors="coerce")
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"].replace("-", "0"), errors="coerce")
    return df

def try_read_tabular(path: Path) -> Optional[pd.DataFrame]:
    # tenta separadores comuns
    seps = [",", ";", "\t", "|"]
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            # heurística: ao menos 2 colunas para considerar tabular
            if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
                # adiciona numeração de linha (se inexistente)
                if "line_number" not in df.columns:
                    df = df.reset_index().rename(columns={"index": "line_number"})
                    df["line_number"] += 1
                return df
        except Exception:
            continue
    return None

TIMESTAMP_CANDIDATE_RES = [
    re.compile(r'\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+\-]\d{4})\]'),  # Apache
    re.compile(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?)'),  # ISO8601
    re.compile(r'(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})'),  # YYYY-MM-DD HH:MM:SS
]

def extract_timestamp_from_text(s: str) -> Optional[pd.Timestamp]:
    for rx in TIMESTAMP_CANDIDATE_RES:
        m = rx.search(s)
        if m:
            txt = m.group(1) if m.lastindex else m.group(0)
            try:
                return pd.to_datetime(txt, utc=True, errors="coerce")
            except Exception:
                continue
    return None

def parse_fallback_text(path: Path) -> pd.DataFrame:
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f, 1):
            s = line.rstrip('\n')
            if not s:
                continue
            ts = extract_timestamp_from_text(s)
            rows.append({"line_number": i, "message": s, "timestamp": ts})
    return pd.DataFrame(rows)

# --------- Orquestrador ---------
def parse_log_file(path: Path) -> pd.DataFrame:
    # "sniff" primeiras 5 linhas
    head_lines = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            if line.strip():
                head_lines.append(line.strip())

    # detectores rápidos
    if any(l.startswith("{") for l in head_lines):
        df = parse_json_lines(path)
        if df is not None:
            df["format"] = "jsonl"
            return df

    if any("=" in l for l in head_lines):
        df = parse_kv_pairs(path)
        if df is not None:
            df["format"] = "kv"
            return df

    # Apache/Nginx
    df = parse_apache_like(path)
    if df is not None:
        df["format"] = "apache"
        return df

    # Tabular genérico
    df = try_read_tabular(path)
    if df is not None:
        df["format"] = "tabular"
        return df

    # Fallback texto livre
    df = parse_fallback_text(path)
    df["format"] = "text"
    return df

def load_logs_to_dataframes(
    folder: str,
    pattern: str = "*.log",
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Lê todos os arquivos que batem com `pattern` em `folder`,
    devolve (por_arquivo, consolidado).
    """
    folder_path = Path(folder)
    files = sorted(folder_path.rglob(pattern))
    per_file: Dict[str, pd.DataFrame] = {}
    all_rows: List[pd.DataFrame] = []

    for p in files:
        try:
            df = parse_log_file(p)
            df.insert(0, "source_file", str(p))
            per_file[str(p)] = df
            all_rows.append(df)
        except Exception as e:
            # registra erro como linha única, para rastrear
            per_file[str(p)] = pd.DataFrame(
                [{"source_file": str(p), "error": str(e)}]
            )

    consolidated = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    # tenta normalizar timestamp, se existir
    for col in ["timestamp", "time", "datetime", "date"]:
        if col in consolidated.columns:
            consolidated["timestamp_norm"] = pd.to_datetime(
                consolidated[col], errors="coerce", utc=True
            )
            break

    return per_file, consolidated

# --------- Exemplo de uso ----------
if __name__ == "__main__":
    # Exemplo: python script.py
    per_file, consolidated = load_logs_to_dataframes("./logs_dir", "*.log")

    # Agora você tem:
    # - per_file: dict { caminho_do_arquivo -> DataFrame }
    # - consolidated: DataFrame com todos os logs unificados

    # Exemplo rápido de inspeção:
    print(f"Arquivos lidos: {len(per_file)}")
    print("Colunas do consolidado:", list(consolidated.columns))
    print(consolidado.head())
