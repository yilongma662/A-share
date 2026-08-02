"""端到端回测验证。

目的是**验证系统链路正确**，不是得出策略结论。样本与已知偏差在报告末尾如实列出。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from gvs import config
from gvs.backtest.engine import run_backtest
from gvs.backtest.metrics import quantile_monotonicity
from gvs.config import BacktestConfig
from gvs.factors.growth import composite_score
from gvs.factors.pit import as_of_snapshot, assert_no_lookahead, build_pit_panel
from gvs.pipeline import build_price_panel
from gvs.storage import Store

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

DATASET = "bars_yahoo"
TOP_N = 15
START = "2019-01-01"


def load_financials(store: Store) -> pd.DataFrame:
    files = sorted((config.CURATED_DIR / "financials").glob("*.parquet"))
    frames = [store.read("financials", f.stem) for f in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("无财务数据，请先执行 scripts/fetch_sample.py")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    store = Store()
    codes = sorted(p.stem for p in (config.CURATED_DIR / DATASET).glob("*.parquet"))
    prices = build_price_panel(codes, store, dataset=DATASET)
    fin = load_financials(store)

    print(f"标的 {len(codes)} 只   价格面板 {prices.shape}   财务记录 {len(fin)} 条")
    print(f"价格区间 {prices.index[0]:%Y-%m-%d} ~ {prices.index[-1]:%Y-%m-%d}\n")

    cfg = BacktestConfig(start=START, rebalance="M", top_n=TOP_N)

    # 每个调仓日只用当日已公告的财报打分，selector 内部不接触未来数据
    score_cache: dict[pd.Timestamp, list[str]] = {}

    def selector(as_of: pd.Timestamp, available: list[str]) -> list[str]:
        if as_of not in score_cache:
            snap = as_of_snapshot(fin, as_of)
            if snap.empty:
                score_cache[as_of] = []
            else:
                assert_no_lookahead(snap.assign(_as_of=as_of))
                scored = composite_score(snap)
                score_cache[as_of] = scored.dropna(subset=["score_total"])["code"].tolist()
        ranked = score_cache[as_of]
        return [c for c in ranked if c in available][:TOP_N]

    result = run_backtest(prices, selector, cfg)

    # 等权持有全部标的作为参照。样本本身就是当前大市值股，参照系已含幸存者偏差
    def hold_all(as_of, available):
        return list(available)

    bench = run_backtest(prices, hold_all, cfg)

    print("=" * 60)
    print("成长因子选股（每月调仓，等权 Top %d）" % TOP_N)
    print("=" * 60)
    print(result.summary())
    print()
    print("=" * 60)
    print("参照组：等权持有全部样本股")
    print("=" * 60)
    print(bench.summary())

    excess = result.performance.annual_return - bench.performance.annual_return
    print(f"\n年化超额  {excess:+.2%}")

    # 分组单调性：因子是否真实有效的核心证据
    print("\n" + "=" * 60)
    print("分组单调性检验（下一调仓期收益）")
    print("=" * 60)
    rebal = sorted(score_cache)
    rows = []
    skipped = 0
    for i, d in enumerate(rebal[:-1]):
        snap = as_of_snapshot(fin, d)
        if snap.empty:
            skipped += 1
            continue
        scored = composite_score(snap).dropna(subset=["score_total"])
        nxt = rebal[i + 1]
        # 不能用 DataFrame.asof：它要求整行无 NaN，样本含次新股时每行都有 NaN，
        # 会返回全空。改为逐列取截至该日的最后一个有效价。
        p0 = prices.loc[:d].ffill().iloc[-1]
        p1 = prices.loc[:nxt].ffill().iloc[-1]
        fwd = (p1 / p0 - 1).replace([float("inf"), float("-inf")], pd.NA).dropna()
        common = [c for c in scored["code"] if c in fwd.index]
        if len(common) < 25:
            skipped += 1
            continue
        s = scored.set_index("code").loc[common, "score_total"]
        rows.append(pd.DataFrame({"score": s, "fwd": fwd.loc[common], "date": d}))
    if skipped:
        print(f"（{skipped}/{len(rebal) - 1} 期因样本不足被跳过）")

    if rows:
        pooled = pd.concat(rows)
        table = quantile_monotonicity(pooled["score"], pooled["fwd"], n_groups=5)
        print(table.to_string())
        spread = table.loc["Q5", "mean"] - table.loc["Q1", "mean"]
        ranks = table["mean"].tolist()
        monotonic = all(a <= b for a, b in zip(ranks, ranks[1:]))
        print(f"\nQ5-Q1 收益差  {spread:+.2%}   单调递增: {'是' if monotonic else '否'}")
        if not monotonic:
            print("分组收益不单调 —— 该因子与收益无稳定关系，不具备可用性。")
    else:
        print("样本不足，无法进行分组检验。")

    print("\n" + "=" * 60)
    print("本次回测的已知缺陷（结论不可用于实盘）")
    print("=" * 60)
    for line in (
        f"样本仅 {len(codes)} 只，且按**当前**总市值选取，同时含幸存者偏差与选样前视偏差",
        "无退市股，历史上归零的公司完全缺席，收益被系统性高估",
        "未处理涨跌停：一字板实际无法成交，回测按可成交处理",
        f"行情来自 Yahoo（不复权口径），除权日会产生虚假跌幅，长周期回测失真",
        "因子权重为未经优化的初始假设，未做样本内外划分",
    ):
        print(f"  · {line}")

    out = config.REPORT_DIR / "backtest_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "strategy": result.performance.to_dict(),
        "benchmark": bench.performance.to_dict(),
        "codes": len(codes), "top_n": TOP_N, "start": START,
        "dataset": DATASET, "trades": result.trades,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已归档 {out}")


if __name__ == "__main__":
    main()
