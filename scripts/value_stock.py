"""个股估值分析。

    python3 scripts/value_stock.py 002185 --peers 600584 002156 603005 688362 688403 688352
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from gvs import config
from gvs.research.valuation import (
    Scenario,
    ValuationResult,
    build_multiples,
    fair_value_range,
    gather_inputs,
    implied_roe,
    pb_roe_cross_section,
    pb_to_roe_ratio,
    peer_comparison,
    percentile_of,
    ttm_from_cumulative,
)
from gvs.storage import Store

logging.basicConfig(level=logging.ERROR)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

DATASET = "bars_yahoo"


def load(code: str, store: Store) -> tuple[pd.DataFrame, pd.Series]:
    fin = store.read("financials", code)
    bars = store.read(DATASET, code)
    px = (bars.set_index(pd.to_datetime(bars["date"]))["close"]
          if not bars.empty else pd.Series(dtype=float))
    return fin, px


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--peers", nargs="*", default=[])
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args()

    store = Store()
    fin, px = load(args.code, store)
    if fin.empty or px.empty:
        raise SystemExit(f"{args.code} 本地无数据，请先 python -m gvs.cli fetch {args.code}")

    as_of = pd.Timestamp(args.as_of) if args.as_of else px.index[-1]
    px = px[px.index <= as_of]
    v = gather_inputs(args.code, fin, float(px.iloc[-1]), as_of)

    print("=" * 74)
    print(f"{v.name} ({v.code})  估值分析   截至 {as_of:%Y-%m-%d}")
    print("=" * 74)
    print(f"收盘价       {v.price:.2f}")
    print(f"总股本       {v.total_share / 1e8:.2f} 亿股")
    print(f"总市值       {v.market_cap / 1e8:.2f} 亿元")
    print(f"每股净资产   {v.bps:.4f} 元   归母净资产 {v.equity / 1e8:.2f} 亿元")
    print(f"TTM 归母净利 {v.ttm_profit / 1e8:.3f} 亿元"
          f"   （依据 {v.report_name}，公告于 {v.notice_date:%Y-%m-%d}）")
    print()
    print(f"PE(TTM)      {v.pe_ttm:.2f}")
    print(f"PB           {v.pb:.2f}")
    print(f"ROE(TTM)     {v.roe_ttm:.2f}%")

    # ── 历史估值分位 ────────────────────────────────────
    hist = build_multiples(px, fin)
    print("\n" + "=" * 74)
    print("历史估值分位")
    print("=" * 74)
    if hist.empty:
        print("无法构建历史估值序列。")
    else:
        for col, cur, label in (("pe_ttm", v.pe_ttm, "PE(TTM)"),
                                ("pb", v.pb, "PB"),
                                ("roe_ttm", v.roe_ttm, "ROE(TTM)%")):
            s = hist[col].dropna()
            if s.empty or cur is None:
                print(f"{label:<12} 无有效历史序列")
                continue
            pct = percentile_of(s, cur)
            print(f"{label:<12} 当前 {cur:>8.2f}   "
                  f"历史分位 {pct:>5.1%}   "
                  f"中位 {s.median():>7.2f}   "
                  f"区间 {s.min():.2f} ~ {s.max():.2f}   "
                  f"样本 {len(s)} 日 ({s.index[0]:%Y-%m}~{s.index[-1]:%Y-%m})")

    # ── 市场隐含预期 ────────────────────────────────────
    print("\n" + "=" * 74)
    print("市场隐含预期（剩余收益模型反推）")
    print("=" * 74)
    print("模型 PB = (ROE - g) / (k - g)，反推当前 PB 隐含的永续 ROE。")
    print()
    print(f"{'折现率 k':<12}{'永续增长 g':<14}{'隐含永续ROE':>14}")
    for k in (8.0, 10.0, 12.0):
        for g in (2.0, 3.0):
            print(f"{k:<12.0f}{g:<14.1f}{implied_roe(v.pb, k, g):>13.1f}%")

    roe_hist = ttm_roe_history(fin)
    if not roe_hist.empty:
        print(f"\n公司历史 TTM ROE：最高 {roe_hist.max():.2f}%  "
              f"中位 {roe_hist.median():.2f}%  最新 {roe_hist.iloc[-1]:.2f}%  "
              f"（{len(roe_hist)} 期）")

    # ── 情景估值 ────────────────────────────────────────
    scenarios = [
        Scenario("悲观", roe=4.5, cost_of_equity=10.0, growth=3.0,
                 note="维持当前盈利水平"),
        Scenario("中性", roe=8.0, cost_of_equity=10.0, growth=3.0,
                 note="回到历史中枢偏上"),
        Scenario("乐观", roe=12.0, cost_of_equity=10.0, growth=3.0,
                 note="接近历史峰值并维持"),
    ]
    result = ValuationResult(inputs=v, scenarios=scenarios, history=hist)
    print("\n" + "=" * 74)
    print("情景估值（假设已显式列出，可自行替换）")
    print("=" * 74)
    tbl = result.scenario_table()
    tbl["合理PB"] = tbl["合理PB"].round(2)
    tbl["目标价"] = tbl["目标价"].round(2)
    tbl["相对现价"] = (tbl["相对现价"] * 100).round(1).astype(str) + "%"
    print(tbl.to_string(index=False))

    print("\n敏感性：目标价对稳态 ROE 与折现率（g=3%）")
    sens = result.sensitivity([4.5, 6.0, 8.0, 10.0, 12.0, 15.0], [8.0, 10.0, 12.0])
    print(sens.round(2).to_string())

    # ── 同业对比 ────────────────────────────────────────
    if args.peers:
        fins = {args.code: fin}
        pxs = {args.code: px}
        for p in args.peers:
            f2, p2 = load(p, store)
            fins[p], pxs[p] = f2, p2
        print("\n" + "=" * 74)
        print("同业横向对比")
        print("=" * 74)
        peers = peer_comparison([args.code] + args.peers, fins, pxs, as_of)
        num = ["价格", "市值(亿)", "PE(TTM)", "PB", "ROE(TTM)%",
               "毛利率%", "净利率%", "营收同比%", "负债率%"]
        for c in num:
            if c in peers:
                peers[c] = pd.to_numeric(peers[c], errors="coerce").round(2)
        print(peers.to_string(index=False))
        result.peers = peers

        cs = pb_roe_cross_section(peers, args.code)
        print("\n" + "=" * 74)
        print("PB-ROE 横截面回归（区分个股错价与行业整体溢价）")
        print("=" * 74)
        if not cs["usable"]:
            print(f"拟合  PB = {cs.get('slope', float('nan')):.3f} × ROE "
                  f"+ {cs.get('intercept', float('nan')):.3f}"
                  f"    n={cs['n']}   R²={cs.get('r2', float('nan')):.3f}")
            print(f"\n结论：{cs['reason']}。")

            ratio = pb_to_roe_ratio(peers, args.code)
            if not ratio.empty:
                print("\n改用 PB/ROE 比值（不依赖线性假设，越低越便宜）：")
                t = ratio[["code", "name", "PB", "ROE(TTM)%", "PB/ROE", "相对中位数"]].copy()
                t["PB/ROE"] = t["PB/ROE"].round(3)
                t["相对中位数"] = (t["相对中位数"] * 100).round(1).astype(str) + "%"
                print(t.to_string(index=False))
                tgt = ratio[ratio["_is_target"]]
                if not tgt.empty:
                    dev = float(tgt["相对中位数"].iloc[0])
                    rank = int((ratio["PB/ROE"] < tgt["PB/ROE"].iloc[0]).sum()) + 1
                    print(f"\n{args.code} PB/ROE = {tgt['PB/ROE'].iloc[0]:.3f}，"
                          f"在 {len(ratio)} 家中排第 {rank} 便宜，"
                          f"较行业中位数 {dev:+.1%}")
        else:
            print(f"拟合  PB = {cs['slope']:.3f} × ROE + {cs['intercept']:.3f}"
                  f"    n={cs['n']}   R²={cs['r2']:.3f}")
            t = cs["table"][["code", "name", "ROE(TTM)%", "PB", "拟合PB", "残差", "相对偏离"]].copy()
            t["拟合PB"] = t["拟合PB"].round(2)
            t["残差"] = t["残差"].round(2)
            t["相对偏离"] = (t["相对偏离"] * 100).round(1).astype(str) + "%"
            print(t.to_string(index=False))
            dev = cs["target_deviation"]
            if dev is not None:
                verdict = ("高于同业拟合值" if dev > 0.1 else
                           "低于同业拟合值" if dev < -0.1 else "与同业拟合值基本一致")
                print(f"\n{args.code} 相对同业 {verdict}（偏离 {dev:+.1%}，"
                      f"拟合 PB {cs['target_fitted_pb']:.2f} vs 实际 {peers.loc[peers['code'] == args.code, 'PB'].iloc[0]:.2f}）")
            if cs["r2"] < 0.5:
                print(f"注意：R²={cs['r2']:.3f} 偏低，ROE 对 PB 的解释力不足，"
                      "该结论可信度有限。")

    print("\n" + "=" * 74)
    print("合理价区间汇总")
    print("=" * 74)
    fv = fair_value_range(v, fin, result.peers if args.peers else None)
    if fv.empty:
        print("数据不足，无法汇总。")
    else:
        t = fv.copy()
        t["合理PB"] = t["合理PB"].round(2)
        t["合理价"] = t["合理价"].round(2)
        t["相对现价"] = (t["相对现价"] * 100).round(1).astype(str) + "%"
        print(t.to_string(index=False))
        lo, hi = fv["合理价"].min(), fv["合理价"].max()
        print(f"\n合理价区间 {lo:.2f} ~ {hi:.2f} 元   现价 {v.price:.2f} 元")
        if hi / lo > 2:
            print("各锚定方法分歧超过一倍 —— 不存在可靠的单一目标价，"
                  "分歧本身即是结论：基本面与同业定价给出不同答案。")

    print("\n" + "=" * 74)
    print("本估值的局限")
    print("=" * 74)
    for line in (
        "剩余收益模型假设 ROE 永续稳定，对强周期行业是明显简化",
        "折现率与永续增长为外生假设，未做 CAPM 推导",
        "行情来自 Yahoo 不复权序列，历史分位未考虑除权对市值口径的影响",
        "同业口径未做会计政策差异调整",
        "未纳入在建工程、产能投放节奏、客户集中度等定性因素",
    ):
        print(f"  · {line}")
    print("\n本报告为数据陈述与情景推演，不构成投资建议。")


def ttm_roe_history(fin: pd.DataFrame) -> pd.Series:
    """历史 TTM ROE 序列（%）。"""
    ttm = ttm_from_cumulative(fin, "net_profit")
    f = fin.copy()
    f["report_date"] = pd.to_datetime(f["report_date"])
    f = f.set_index("report_date")
    equity = f["bps"] * f["total_share"]
    return (ttm / equity.reindex(ttm.index) * 100).dropna()


if __name__ == "__main__":
    main()
