"""本地数据存储。

宪章第一条要求任何数据可溯源，因此 write() 强制注入三个溯源列，
不提供关闭开关 —— 能绕过的约束等于没有约束。
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gvs import config

PROVENANCE_COLS = ("_source", "_endpoint", "_fetched_at")


@dataclass
class Store:
    root: Path = config.CURATED_DIR

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, dataset: str, key: str | None = None) -> Path:
        d = self.root / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.parquet" if key else d / "_all.parquet"

    def write(
        self,
        df: pd.DataFrame,
        dataset: str,
        key: str | None = None,
        *,
        source: str,
        endpoint: str,
        dedup_on: list[str] | None = None,
        sort_on: list[str] | None = None,
    ) -> Path:
        """落盘并注入溯源信息。已存在时做增量合并。"""
        if df.empty:
            raise ValueError(f"拒绝写入空数据集: {dataset}/{key}")

        out = df.copy()
        out["_source"] = source
        out["_endpoint"] = endpoint
        out["_fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

        target = self.path(dataset, key)
        if target.exists():
            old = pd.read_parquet(target)
            out = pd.concat([old, out], ignore_index=True)
            if dedup_on:
                # 保留后写入的记录：财报存在追溯调整，新版本应覆盖旧版本
                out = out.drop_duplicates(subset=dedup_on, keep="last")
        if sort_on:
            out = out.sort_values(sort_on).reset_index(drop=True)

        out.to_parquet(target, index=False)
        self._write_meta(dataset, key, out, source, endpoint)
        return target

    def read(self, dataset: str, key: str | None = None) -> pd.DataFrame:
        target = self.path(dataset, key)
        if not target.exists():
            return pd.DataFrame()
        return pd.read_parquet(target)

    def exists(self, dataset: str, key: str | None = None) -> bool:
        return self.path(dataset, key).exists()

    def last_date(self, dataset: str, key: str, col: str = "date") -> pd.Timestamp | None:
        """用于增量更新：返回已有数据的最新日期。"""
        df = self.read(dataset, key)
        if df.empty or col not in df:
            return None
        return pd.to_datetime(df[col]).max()

    def _write_meta(
        self, dataset: str, key: str | None, df: pd.DataFrame, source: str, endpoint: str
    ) -> None:
        meta = {
            "dataset": dataset,
            "key": key,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "source": source,
            "endpoint": endpoint,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        meta_path = self.path(dataset, key).with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
