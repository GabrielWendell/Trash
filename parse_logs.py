import json
from typing import List, Dict, Tuple, Literal, Optional
import pandas as pd

def load_data(
    file_path: str,
    on_error: Literal["raise", "skip", "collect"] = "skip",
    encoding: str = "utf-8",
) -> pd.DataFrame | Tuple[pd.DataFrame, List[Tuple[int, str]]]:
    """
    Lê um arquivo de logs no formato JSON Lines (um JSON por linha) e
    retorna um pandas.DataFrame.

    Parâmetros
    ----------
    file_path : str
        Caminho para o arquivo .log / .jsonl.
    on_error : {"raise", "skip", "collect"}, default "skip"
        - "raise": lança erro ao encontrar linha inválida
        - "skip": ignora linhas inválidas
        - "collect": coleta linhas inválidas e retorna (df, bad_lines)
    encoding : str, default "utf-8"
        Codificação usada na leitura do arquivo.

    Retorno
    -------
    pd.DataFrame
        DataFrame com os objetos JSON (dict) por linha.
        Se on_error="collect", retorna (DataFrame, bad_lines),
        onde bad_lines é uma lista de tuplas (line_number, line_text).
    """
    rows: List[Dict] = []
    bad_lines: List[Tuple[int, str]] = []

    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue  # pula linhas vazias
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                if on_error == "raise":
                    raise
                elif on_error == "collect":
                    bad_lines.append((i, s))
                # on_error == "skip": apenas ignora
                continue

            # Se a linha não for um dict (por ex., lista ou número), guarda como "value"
            if not isinstance(obj, dict):
                obj = {"value": obj}
            obj["__line__"] = i
            rows.append(obj)

    df = pd.DataFrame(rows)

    if on_error == "collect":
        return df, bad_lines
    return df
