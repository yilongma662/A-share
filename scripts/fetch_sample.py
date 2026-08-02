"""拉取回测样本股。

样本选取方式与其偏差在 reports/ 的回测报告中必须如实说明：
本脚本按**当前**总市值取样，因此样本本身带有幸存者偏差与前视偏差
（今日的大市值公司在十年前未必存在或未必是大市值）。
该样本仅用于验证系统链路，不可用于得出策略结论。
"""
from __future__ import annotations

import logging
import sys

import pandas as pd

from gvs import config, pipeline
from gvs.datasource import EastmoneyClient, PriceService
from gvs.storage import Store

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 80


def pick_sample(n: int) -> pd.DataFrame:
    store = Store()
    files = sorted((config.CURATED_DIR / "universe").glob("*.parquet"))
    if not files:
        raise SystemExit("无 universe 数据，请先执行 python -m gvs.cli universe")
    uni = store.read("universe", files[-1].stem)

    pool = uni[
        (~uni["is_st"])
        & (uni["total_mv"] > 5e9)          # 50 亿以上，规避流动性不足
        & (uni["board"] != "北交所")        # 北交所流动性与规则差异大，单独研究
        & uni["pe_ttm"].notna()
    ].copy()

    # 按板块分层取样，避免样本集中于单一板块
    per_board = max(n // pool["board"].nunique(), 1)
    picked = (pool.sort_values("total_mv", ascending=False)
                  .groupby("board", group_keys=False)
                  .head(per_board))
    return picked.head(n)


def main() -> None:
    config.ensure_dirs()
    sample = pick_sample(N)
    print(f"样本 {len(sample)} 只：")
    print(sample.groupby("board").size().to_string())
    print()

    client = EastmoneyClient()
    prices = PriceService(eastmoney=client)
    store = Store()
    ok = failed = 0
    for i, row in enumerate(sample.itertuples(), 1):
        code = row.code
        try:
            bars = pipeline.ingest_bars(code, client, store, prices=prices)
            fin = pipeline.ingest_financials(code, client, store)
            print(f"[{i:>3}/{len(sample)}] {code} {row.name:<8} "
                  f"行情{len(bars):>5} 财务{len(fin):>3}", flush=True)
            ok += 1
        except Exception as exc:
            print(f"[{i:>3}/{len(sample)}] {code} 失败: {exc}", flush=True)
            failed += 1

    print(f"\n完成 {ok} 只，失败 {failed} 只")
    if failed:
        print("失败标的已显式记录 —— 若直接忽略会造成隐性幸存者偏差")


if __name__ == "__main__":
    main()
