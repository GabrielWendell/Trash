#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Tuple, Literal
import argparse
import pandas as pd

# ============ load_data (JSON Lines) ============
def load_data(
    file_path: str,
    on_error: Literal["raise", "skip", "collect"] = "skip",
    encoding: str = "utf-8",
) -> pd.DataFrame | Tuple[pd.DataFrame, List[Tuple[int, str]]]:
    rows: List[Dict] = []
    bad_lines: List[Tuple[int, str]] = []

    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                if on_error == "raise":
                    raise
                elif on_error == "collect":
                    bad_lines.append((i, s))
                continue

            if not isinstance(obj, dict):
                obj = {"value": obj}
            obj["__line__"] = i
            rows.append(obj)

    df = pd.DataFrame(rows)
    if on_error == "collect":
        return df, bad_lines
    return df


# ============ utilidades ============
def safe_rel_name(base_dir: Path, file_path: Path) -> str:
    """
    Constrói um nome de saída estável a partir do caminho relativo,
    trocando separadores por '__' e a extensão por '.csv'
    """
    rel = file_path.relative_to(base_dir)
    no_ext = rel.as_posix().replace("/", "__").rsplit(".", 1)[0]
    return f"{no_ext}.csv"


def process_logs(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.log",
    recursive: bool = True,
    encoding: str = "utf-8",
    on_error: Literal["raise", "skip", "collect"] = "collect",
    write_consolidated: bool = True,
    verbose: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Descoberta dos arquivos
    files = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    if verbose:
        print(f"[info] arquivos .log encontrados: {len(files)}")

    all_dfs: List[pd.DataFrame] = []
    all_bad: List[Dict] = []

    for fp in files:
        if verbose:
            print(f"[info] lendo: {fp}")

        if on_error == "collect":
            df, bad_lines = load_data(str(fp), on_error="collect", encoding=encoding)
        else:
            df = load_data(str(fp), on_error=on_error, encoding=encoding)
            bad_lines = []

        # adiciona metadados úteis
        if not df.empty:
            df.insert(0, "__source_file__", str(fp))
            all_dfs.append(df)

        # salva CSV por arquivo
        out_name = safe_rel_name(input_dir, fp)
        out_path = output_dir / out_name
        if not df.empty:
            df.to_csv(out_path, index=False)
            if verbose:
                print(f"[ok] salvo: {out_path} (linhas: {len(df)})")
        else:
            # cria um CSV vazio com só cabeçalho mínimo
            pd.DataFrame(columns=["__source_file__", "__line__"]).to_csv(out_path, index=False)
            if verbose:
                print(f"[warn] {fp} não gerou linhas válidas; CSV vazio salvo: {out_path}")

        # acumula linhas ruins (se houver)
        for ln, txt in bad_lines:
            all_bad.append(
                {"__source_file__": str(fp), "__line__": ln, "__raw__": txt}
            )

    # Consolidado
    if write_consolidated:
        consolidated_path = output_dir / "_consolidated.csv"
        if all_dfs:
            big = pd.concat(all_dfs, ignore_index=True)
            big.to_csv(consolidated_path, index=False)
            if verbose:
                print(f"[ok] consolidado salvo: {consolidated_path} (linhas: {len(big)})")
        else:
            pd.DataFrame().to_csv(consolidated_path, index=False)
            if verbose:
                print(f"[warn] nenhum dado válido para consolidado; criado vazio: {consolidated_path}")

    # Relatório de linhas inválidas
    if on_error == "collect":
        bad_path = output_dir / "_bad_lines.csv"
        if all_bad:
            pd.DataFrame(all_bad).to_csv(bad_path, index=False)
            if verbose:
                print(f"[ok] relatório de linhas inválidas: {bad_path} (total: {len(all_bad)})")
        else:
            # ainda assim criamos um arquivo vazio para registro
            pd.DataFrame(columns=["__source_file__", "__line__", "__raw__"]).to_csv(bad_path, index=False)
            if verbose:
                print(f"[info] nenhuma linha inválida detectada; criado vazio: {bad_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Converte arquivos .log (JSON Lines) em CSVs com pandas."
    )
    parser.add_argument("--input-dir", default="logs_dir", type=str, help="Pasta de entrada com .log")
    parser.add_argument("--output-dir", default="logs_csv", type=str, help="Pasta para salvar CSVs")
    parser.add_argument("--pattern", default="*.log", type=str, help="Glob pattern (ex.: *.log)")
    parser.add_argument("--no-recursive", action="store_true", help="Não buscar recursivamente")
    parser.add_argument("--encoding", default="utf-8", type=str, help="Encoding de leitura (default: utf-8)")
    parser.add_argument(
        "--on-error",
        choices=["raise", "skip", "collect"],
        default="collect",
        help="Comportamento para linhas inválidas (default: collect)",
    )
    parser.add_argument("--no-consolidated", action="store_true", help="Não gerar _consolidated.csv")
    parser.add_argument("--quiet", action="store_true", help="Menos logs na saída")

    args = parser.parse_args()

    process_logs(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        pattern=args.pattern,
        recursive=not args.no_recursive,
        encoding=args.encoding,
        on_error=args.on_error,
        write_consolidated=not args.no_consolidated,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
