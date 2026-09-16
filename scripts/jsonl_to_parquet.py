"""Turn builder jsonl into verl parquet. Does not import verl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    args = p.parse_args()
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pip install pandas pyarrow") from exc
    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_rows(src)).to_parquet(dst, index=False)
    print(f"wrote {dst} n={sum(1 for _ in src.open(encoding='utf-8') if _.strip())}")


if __name__ == "__main__":
    main()
