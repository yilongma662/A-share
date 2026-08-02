"""命令行入口。

    python -m gvs.cli health                    检查数据源可用性
    python -m gvs.cli diagnose 002185           个股诊断
    python -m gvs.cli fetch 002185 600519       拉取并落盘
    python -m gvs.cli universe                  更新全市场标的表
    python -m gvs.cli screen --top 20           成长股筛选（需先 fetch）
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from gvs import config, pipeline
from gvs.datasource import EastmoneyClient, PriceService
from gvs.factors.growth import composite_score
from gvs.factors.pit import as_of_snapshot
from gvs.research.diagnose import diagnose
from gvs.research.valuation import (
    build_multiples,
    gather_inputs,
    implied_roe,
    percentile_of,
)
from gvs.storage import Store


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def cmd_health(args) -> int:
    status = PriceService().health_check()
    for name, ok in status.items():
        print(f"  {name:<12} {'可用' if ok else '不可用'}")
    if not any(status.values()):
        print("\n所有行情源均不可用，无法继续。东财对境外 IP 有累积限流，通常数分钟后恢复。")
        return 1
    if not status.get("eastmoney"):
        print("\n东财不可用，将降级至 Yahoo。注意复权口径不同，不可与东财序列混用。")
    return 0


def cmd_diagnose(args) -> int:
    for code in args.codes:
        try:
            print(diagnose(code, as_of=args.as_of).report())
        except Exception as exc:
            print(f"{code} 诊断失败: {exc}", file=sys.stderr)
            return 1
        print()
    return 0


def cmd_fetch(args) -> int:
    config.ensure_dirs()
    client = EastmoneyClient()
    store = Store()
    ok = failed = 0
    for code in args.codes:
        try:
            bars = pipeline.ingest_bars(code, client, store, adjust=args.adjust)
            fin = pipeline.ingest_financials(code, client, store)
            print(f"  {code}  行情 {len(bars):>5} 根   财务 {len(fin):>3} 期")
            ok += 1
        except Exception as exc:
            # 失败必须显式记录：静默跳过会造成隐性幸存者偏差
            print(f"  {code}  失败: {exc}", file=sys.stderr)
            failed += 1
    print(f"\n完成 {ok} 只，失败 {failed} 只")
    return 1 if failed and not ok else 0


def cmd_universe(args) -> int:
    config.ensure_dirs()
    df = pipeline.ingest_universe()
    print(f"全市场 {len(df)} 只标的")
    print(df.groupby("board").size().to_string())
    print(f"\nST 标的 {int(df['is_st'].sum())} 只")
    return 0


def cmd_value(args) -> int:
    store = Store()
    fin = store.read("financials", args.code)
    bars = store.read("bars_yahoo", args.code)
    if bars.empty:
        bars = store.read("bars_fq0", args.code)
    if fin.empty or bars.empty:
        print(f"{args.code} 本地无数据，请先 python -m gvs.cli fetch {args.code}",
              file=sys.stderr)
        return 1

    px = bars.set_index(pd.to_datetime(bars["date"]))["close"]
    as_of = pd.Timestamp(args.as_of) if args.as_of else px.index[-1]
    px = px[px.index <= as_of]
    v = gather_inputs(args.code, fin, float(px.iloc[-1]), as_of)

    print(f"{v.name} ({v.code})   截至 {as_of:%Y-%m-%d}   收盘 {v.price:.2f}")
    print(f"  总市值      {v.market_cap / 1e8:>10.2f} 亿")
    print(f"  TTM归母净利 {v.ttm_profit / 1e8:>10.3f} 亿   "
          f"（{v.report_name}，公告 {v.notice_date:%Y-%m-%d}）")
    print(f"  PE(TTM)     {v.pe_ttm:>10.2f}")
    print(f"  PB          {v.pb:>10.2f}")
    print(f"  ROE(TTM)    {v.roe_ttm:>10.2f}%")

    hist = build_multiples(px, fin)
    if not hist.empty:
        print("\n  历史分位：", end="")
        parts = []
        for col, cur, label in (("pe_ttm", v.pe_ttm, "PE"), ("pb", v.pb, "PB")):
            s = hist[col].dropna()
            if not s.empty and cur is not None:
                parts.append(f"{label} {percentile_of(s, cur):.0%}")
        print("   ".join(parts))

    print(f"\n  当前 PB {v.pb:.2f} 隐含的永续 ROE（k=10%, g=3%）："
          f"{implied_roe(v.pb, 10.0, 3.0):.1f}%")
    print("  与公司历史 ROE 对照即可判断该预期是否曾被实现过。")
    print("\n  完整分析：python3 scripts/value_stock.py "
          f"{args.code} --peers <同业代码>")
    return 0


def cmd_screen(args) -> int:
    store = Store()
    fin_dir = config.CURATED_DIR / "financials"
    files = sorted(fin_dir.glob("*.parquet")) if fin_dir.exists() else []
    if not files:
        print("本地无财务数据，请先执行 fetch", file=sys.stderr)
        return 1

    frames = [store.read("financials", f.stem) for f in files]
    fin = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today()

    snap = as_of_snapshot(fin, as_of)
    if snap.empty:
        print(f"{as_of:%Y-%m-%d} 时点无已公告财报", file=sys.stderr)
        return 1

    scored = composite_score(snap)
    dropped = scored.attrs.get("dropped_factors") or []
    cols = [c for c in ("code", "report_name", "revenue_growth",
                        "deducted_profit_growth", "roe", "gross_margin",
                        "score_growth", "score_quality", "score_total") if c in scored]
    pd.set_option("display.width", 200)
    print(f"评估时点 {as_of:%Y-%m-%d}   样本 {len(snap)} 只   权重 {scored.attrs['weights']}")
    if dropped:
        print(f"因覆盖率不足剔除的因子: {', '.join(dropped)}")
    print(scored.head(args.top)[cols].to_string(index=False))
    print("\n注意：本打分未经回测验证，权重为初始假设，不构成选股结论。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gvs", description="GVS Infinity 研究系统")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="检查数据源可用性").set_defaults(func=cmd_health)

    d = sub.add_parser("diagnose", help="个股诊断")
    d.add_argument("codes", nargs="+")
    d.add_argument("--as-of", default=None, help="回溯到指定日期（YYYY-MM-DD）")
    d.set_defaults(func=cmd_diagnose)

    f = sub.add_parser("fetch", help="拉取行情与财务并落盘")
    f.add_argument("codes", nargs="+")
    f.add_argument("--adjust", type=int, default=1, choices=[0, 1, 2],
                   help="0=不复权 1=前复权(回测用) 2=后复权")
    f.set_defaults(func=cmd_fetch)

    sub.add_parser("universe", help="更新全市场标的表").set_defaults(func=cmd_universe)

    val = sub.add_parser("value", help="个股估值速览")
    val.add_argument("code")
    val.add_argument("--as-of", default=None)
    val.set_defaults(func=cmd_value)

    s = sub.add_parser("screen", help="成长股筛选")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--as-of", default=None)
    s.set_defaults(func=cmd_screen)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
