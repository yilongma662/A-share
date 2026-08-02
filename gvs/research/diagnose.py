"""个股诊断：把行情与财务合成一份带证据的结构化判断。

本模块**不给买卖建议**。它只做三件事：陈述数据、标注异常、指出证据不足之处。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gvs.datasource.eastmoney import EastmoneyClient
from gvs.datasource.prices import PriceService
from gvs.factors.pit import as_of_snapshot


@dataclass
class Evidence:
    """一条判断 + 它依据的数据。没有 value 的判断不允许存在。"""

    label: str
    value: str
    verdict: str          # positive / negative / neutral / unknown
    note: str = ""

    def line(self) -> str:
        mark = {"positive": "+", "negative": "-", "neutral": "·", "unknown": "?"}[self.verdict]
        tail = f"  — {self.note}" if self.note else ""
        return f"  [{mark}] {self.label:<16} {self.value}{tail}"


@dataclass
class Diagnosis:
    code: str
    name: str
    as_of: pd.Timestamp
    price: float
    provider: str = "eastmoney"
    technical: list[Evidence] = field(default_factory=list)
    fundamental: list[Evidence] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    def report(self) -> str:
        parts = [
            f"{self.name} ({self.code})   截至 {self.as_of:%Y-%m-%d}   收盘 {self.price:.2f}"
            f"   [行情源: {self.provider}]",
            "=" * 60,
            "技术面",
            *[e.line() for e in self.technical],
            "",
            "基本面",
            *[e.line() for e in self.fundamental],
        ]
        if self.flags:
            parts += ["", "风险标记", *[f"  ! {f}" for f in self.flags]]
        if self.gaps:
            parts += ["", "证据不足", *[f"  ? {g}" for g in self.gaps]]
        parts += ["", "本报告为数据陈述，非投资建议。所有判断依据上列数值，可自行复核。"]
        return "\n".join(parts)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _same_quarter_last_year(fin: pd.DataFrame, row: pd.Series, col: str) -> float | None:
    """取去年同一报告期的指标值。

    必须同季度对比 —— A 股财报为累计口径，一季报与年报的周转天数不可比。
    """
    if col not in fin or "report_date" not in fin:
        return None
    rd = pd.Timestamp(row["report_date"])
    target = rd - pd.DateOffset(years=1)
    match = fin[pd.to_datetime(fin["report_date"]).dt.normalize() == target.normalize()]
    if match.empty:
        return None
    v = match.iloc[-1].get(col)
    return float(v) if pd.notna(v) else None


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    diff = _ema(close, 12) - _ema(close, 26)
    dea = _ema(diff, 9)
    return diff, dea, 2 * (diff - dea)


def diagnose(
    code: str,
    client: EastmoneyClient | None = None,
    as_of: str | pd.Timestamp | None = None,
    prices: PriceService | None = None,
) -> Diagnosis:
    client = client or EastmoneyClient()
    prices = prices or PriceService(eastmoney=client)

    bars = prices.daily_bars(code, adjust=1)
    if bars.empty:
        raise ValueError(f"{code} 无行情数据")
    provider = str(bars["_provider"].iloc[0]) if "_provider" in bars else "unknown"

    try:
        fin = client.financials(code)
    except Exception as exc:  # 财务接口与行情接口限流独立，一方失败不应中断全部分析
        fin = pd.DataFrame()
        _fin_error = str(exc)
    else:
        _fin_error = ""

    if as_of is not None:
        bars = bars[bars["date"] <= pd.Timestamp(as_of)]
    last = bars.iloc[-1]
    asof_ts = pd.Timestamp(last["date"])
    close = bars.set_index("date")["close"]

    name = str(last["name"]) if "name" in bars.columns and pd.notna(last.get("name")) else code
    if name == code and not fin.empty and "name" in fin:
        name = str(fin["name"].iloc[-1])

    d = Diagnosis(code=code, name=name, as_of=asof_ts,
                  price=float(last["close"]), provider=provider)
    if provider == "yahoo":
        d.gaps.append("东财限流，行情来自 Yahoo（不复权口径），均线与历史高点会与东财口径存在差异")
    if _fin_error:
        d.gaps.append(f"财务数据获取失败：{_fin_error}")

    # ── 技术面 ────────────────────────────────────────────
    mas = {n: close.rolling(n).mean().iloc[-1] for n in (5, 10, 20, 60, 120, 250) if len(close) >= n}
    below = [n for n, v in mas.items() if last["close"] < v]
    ma_txt = " ".join(f"MA{n}:{v:.2f}" for n, v in mas.items())
    d.technical.append(Evidence(
        "均线位置", ma_txt,
        "negative" if len(below) >= len(mas) * 0.8 else
        "positive" if not below else "neutral",
        f"低于 {len(below)}/{len(mas)} 条均线",
    ))

    if len(close) >= 60:
        diff, dea, hist = macd(close)
        d.technical.append(Evidence(
            "MACD", f"DIFF {diff.iloc[-1]:.2f}  DEA {dea.iloc[-1]:.2f}  柱 {hist.iloc[-1]:.2f}",
            "negative" if hist.iloc[-1] < 0 and diff.iloc[-1] < 0 else
            "positive" if hist.iloc[-1] > 0 and diff.iloc[-1] > 0 else "neutral",
        ))

    for window, label in ((20, "近20日"), (60, "近60日")):
        if len(close) > window:
            chg = close.iloc[-1] / close.iloc[-window - 1] - 1
            d.technical.append(Evidence(
                f"{label}涨跌", f"{chg:+.2%}",
                "negative" if chg < -0.15 else "positive" if chg > 0.15 else "neutral",
            ))

    hi = close.max()
    span = f"{close.index[0]:%Y-%m}~{close.index[-1]:%Y-%m}"
    d.technical.append(Evidence(
        "距区间高点", f"{last['close'] / hi - 1:+.1%}", "neutral",
        f"高点 {hi:.2f}，取数区间 {span}（{len(close)} 根日线）",
    ))

    # 波动率：年化，用于判断是否属于高波动标的
    if len(close) > 60:
        vol = close.pct_change().tail(60).std() * np.sqrt(243)
        d.technical.append(Evidence(
            "年化波动率(60日)", f"{vol:.1%}",
            "negative" if vol > 0.60 else "neutral",
        ))

    # ── 基本面 ────────────────────────────────────────────
    if fin.empty:
        d.gaps.append("无财务数据，基本面判断缺失")
        return d

    snap = as_of_snapshot(fin, asof_ts)
    if snap.empty:
        d.gaps.append(f"{asof_ts:%Y-%m-%d} 时点无已公告财报")
        return d
    row = snap.iloc[0]

    d.fundamental.append(Evidence(
        "最新报告期", f"{row.get('report_name', '')}  公告于 {row['notice_date']:%Y-%m-%d}",
        "neutral", f"数据滞后 {int(row['_data_lag_days'])} 天",
    ))

    rev_yoy = row.get("revenue_yoy")
    dp_yoy = row.get("net_profit_deducted_yoy")
    np_yoy = row.get("net_profit_yoy")

    if pd.notna(rev_yoy):
        d.fundamental.append(Evidence(
            "营收同比(累计)", f"{rev_yoy:+.1f}%   ({row['revenue'] / 1e8:.2f} 亿)",
            "positive" if rev_yoy > 20 else "negative" if rev_yoy < 0 else "neutral",
        ))
    if pd.notna(dp_yoy):
        d.fundamental.append(Evidence(
            "扣非净利同比", f"{dp_yoy:+.1f}%",
            "positive" if dp_yoy > 20 else "negative" if dp_yoy < 0 else "neutral",
            "扣非口径剔除补贴与资产处置",
        ))
    elif pd.notna(np_yoy):
        d.fundamental.append(Evidence("归母净利同比", f"{np_yoy:+.1f}%", "neutral"))
        d.gaps.append("扣非净利同比缺失，无法排除非经常性损益影响")

    # 单季度口径：累计数具有平滑效应，会掩盖最近一季的转折
    q_rev, q_dp = row.get("q_revenue_yoy"), row.get("q_deducted_yoy")
    if pd.notna(q_rev):
        note = ""
        if pd.notna(rev_yoy) and q_rev < rev_yoy - 5:
            note = "单季弱于累计，增长在减速"
        elif pd.notna(rev_yoy) and q_rev > rev_yoy + 5:
            note = "单季强于累计，增长在加速"
        d.fundamental.append(Evidence(
            "营收同比(单季)", f"{q_rev:+.1f}%",
            "positive" if q_rev > 20 else "negative" if q_rev < 0 else "neutral", note,
        ))
    if pd.notna(q_dp):
        d.fundamental.append(Evidence(
            "扣非净利同比(单季)", f"{q_dp:+.1f}%",
            "positive" if q_dp > 20 else "negative" if q_dp < 0 else "neutral",
        ))

    # 增速趋势：连续 4 期单季营收增速，看方向而非单点
    hist = fin[fin["notice_date"] <= asof_ts].tail(5)
    if "q_revenue_yoy" in hist and hist["q_revenue_yoy"].notna().sum() >= 3:
        seq = hist["q_revenue_yoy"].dropna().tail(4)
        trend = " → ".join(f"{v:+.0f}%" for v in seq)
        rising = seq.iloc[-1] > seq.iloc[0]
        d.fundamental.append(Evidence(
            "增速趋势", trend, "positive" if rising else "negative",
            "单季营收同比，最近 4 期",
        ))

    for key, label, good, bad in (
        ("roe", "ROE(加权)", 10.0, 3.0),
        ("roic", "ROIC", 8.0, 3.0),
        ("gross_margin", "毛利率", 30.0, 15.0),
        ("net_margin", "净利率", 10.0, 2.0),
    ):
        v = row.get(key)
        if pd.notna(v):
            d.fundamental.append(Evidence(
                label, f"{v:.2f}%",
                "positive" if v >= good else "negative" if v < bad else "neutral",
            ))

    dr = row.get("debt_ratio")
    if pd.notna(dr):
        d.fundamental.append(Evidence(
            "资产负债率", f"{dr:.1f}%",
            "negative" if dr > 65 else "positive" if dr < 40 else "neutral",
        ))

    ocf_rev = row.get("ocf_to_revenue")
    if pd.notna(ocf_rev):
        d.fundamental.append(Evidence(
            "经营现金流/营收", f"{ocf_rev:.3f}",
            "positive" if ocf_rev > 0.15 else "negative" if ocf_rev <= 0 else "neutral",
            "衡量收入的现金实现程度",
        ))

    ocf, eps = row.get("ocf_per_share"), row.get("eps")
    if pd.notna(ocf) and pd.notna(eps) and abs(eps) > 1e-9:
        ratio = ocf / eps
        d.fundamental.append(Evidence(
            "现金含量", f"经营现金流/EPS = {ratio:.2f}",
            "positive" if ratio >= 1 else "negative" if ratio < 0.5 else "neutral",
            "低于 1 说明利润缺少现金支撑" if ratio < 1 else "利润有现金支撑",
        ))
    else:
        d.gaps.append("现金流数据不足，无法验证利润质量")

    # 周转天数：与去年同期比才有意义，绝对值高低取决于行业
    recv_days = row.get("receivable_days")
    if pd.notna(recv_days):
        prior = _same_quarter_last_year(fin, row, "receivable_days")
        note, verdict = "", "neutral"
        if prior is not None and prior > 0:
            chg = recv_days / prior - 1
            note = f"去年同期 {prior:.0f} 天，{chg:+.0%}"
            verdict = "negative" if chg > 0.25 else "positive" if chg < -0.15 else "neutral"
        d.fundamental.append(Evidence("应收周转天数", f"{recv_days:.0f} 天", verdict, note))

    inv_days = row.get("inventory_days")
    if pd.notna(inv_days):
        prior = _same_quarter_last_year(fin, row, "inventory_days")
        note, verdict = "", "neutral"
        if prior is not None and prior > 0:
            chg = inv_days / prior - 1
            note = f"去年同期 {prior:.0f} 天，{chg:+.0%}"
            verdict = "negative" if chg > 0.25 else "positive" if chg < -0.15 else "neutral"
        d.fundamental.append(Evidence("存货周转天数", f"{inv_days:.0f} 天", verdict, note))

    rd = row.get("rd_expense")
    if pd.notna(rd) and pd.notna(row.get("revenue")) and row["revenue"] > 0:
        rd_ratio = rd / row["revenue"]
        d.fundamental.append(Evidence(
            "研发费用率", f"{rd_ratio:.1%}   ({rd / 1e8:.2f} 亿)",
            "positive" if rd_ratio > 0.08 else "neutral",
        ))
    else:
        d.gaps.append("研发支出仅年报/中报披露，本期无数据")

    # ── 成长陷阱识别 ──────────────────────────────────────
    # 判断规则来自宪章第四节"高增速 + 低质量 = 陷阱"，阈值为经验假设，尚未经回测检验
    roe_v = row.get("roe")
    is_annual = str(row.get("report_name", "")).endswith("年报")
    roe_floor = 3.0 if is_annual else 1.0   # 季报 ROE 未年化，不能与年报同阈值比较

    if pd.notna(rev_yoy) and pd.notna(roe_v) and rev_yoy > 15 and roe_v < roe_floor:
        d.flags.append(
            f"高增长低回报：营收 {rev_yoy:+.1f}% 但 ROE 仅 {roe_v:.2f}%，"
            "增长未转化为股东回报，需核查是否靠扩产能/加杠杆堆规模"
        )
    if pd.notna(rev_yoy) and pd.notna(dp_yoy) and rev_yoy > 10 and dp_yoy < 0:
        d.flags.append(f"增收不增利：营收 {rev_yoy:+.1f}%，扣非净利 {dp_yoy:+.1f}%")
    gm = row.get("gross_margin")
    if pd.notna(gm) and gm < 15:
        d.flags.append(f"毛利率仅 {gm:.1f}%，缺乏定价权，利润对成本波动高度敏感")
    if pd.notna(ocf) and pd.notna(eps) and eps > 0 and ocf < 0:
        d.flags.append("净利为正但经营现金流为负，需核查应收账款与存货")

    # 增收 + 应收/存货周转显著变慢，是收入质量恶化的典型组合
    if pd.notna(rev_yoy) and rev_yoy > 10:
        for key, label in (("receivable_days", "应收"), ("inventory_days", "存货")):
            cur = row.get(key)
            prior = _same_quarter_last_year(fin, row, key)
            if pd.notna(cur) and prior and prior > 0 and cur / prior - 1 > 0.25:
                d.flags.append(
                    f"营收 {rev_yoy:+.1f}% 增长的同时{label}周转天数由 {prior:.0f} 天升至 "
                    f"{cur:.0f} 天（{cur / prior - 1:+.0%}），增长的现金质量存疑"
                )
    dr_v = row.get("debt_ratio")
    if pd.notna(dr_v) and dr_v > 65:
        d.flags.append(f"资产负债率 {dr_v:.1f}%，财务杠杆偏高，利率与再融资环境敏感")

    d.gaps.append("以上为单期快照，未做同行业横向对比，不足以支撑估值结论")
    d.gaps.append("陷阱识别阈值为经验假设，尚未经回测验证，不应作为独立决策依据")
    return d
