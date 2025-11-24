def parse_log_text_to_dataframe(text: str):
    import json
    import pandas as pd

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    return pd.DataFrame(rows)
