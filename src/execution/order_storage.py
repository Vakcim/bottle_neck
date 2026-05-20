"""CSV storage helpers for order intents and skipped decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd


class Serializable(Protocol):
    def to_dict(self) -> dict: ...


def append_records_csv(path: Path, records: Iterable[Serializable | dict]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in records]
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        out = pd.concat([old_df, new_df], ignore_index=True) if not new_df.empty else old_df
    else:
        out = new_df
    out.to_csv(path, index=False)
    return out
