"""估值模块。

A 股财报为**年内累计**口径（一季报、中报、三季报、年报），直接拿来算 TTM 会错得离谱。
本模块先还原 TTM，再用 point-in-time 对齐生成历史估值序列。

估值方法的选择依据 —— PE 对低利润率、强周期的公司极不稳定（分母趋近于零时 PE 爆炸），
因此本模块同时提供 PB-ROE 框架：在剩余收益模型下，合理 PB 由 ROE 决定，
可以反推「当前股价隐含了多高的 ROE」，这个问题比「PE 多少倍算贵」可证伪得多。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gvs.factors.pit import as_of_snapshot

# 报告期月份 -> 年内第几个累计期
PERIOD_OF_MONTH = {3: 1, 6: 2, 9: 3, 12: 4}


def ttm_from_cumulative(fin: pd.DataFrame, col: str = "net_profit") -> pd.Series:
    """从累计口径报表还原 TTM（滚动十二个月）。

    A 股财报在年内累计：Q1 / 中报 / 三季报 / 年报。因此
        TTM(第 p 期, y 年) = 年报(y-1) - 累计(第 p 期, y-1) + 累计(第 p 期, y)
    年报本身即为完整一年，直接取用。

    缺少上年同期或上年年报时返回 NaN —— 不做近似填补。
    """
    if fin.empty or col not in fin:
        return pd.Series(dtype=float)

    df = fin.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date").drop_duplicates("report_date", keep="last")
    df["year"] = df["report_date"].dt.year
    df["period"] = df["report_date"].dt.month.map(PERIOD_OF_MONTH)
    df = df[df["period"].notna()]

    cum = {(int(r.year), int(r.period)): r._asdict().get(col)
           for r in df.itertuples() if pd.notna(getattr(r, col, None))}
    annual = {y: v for (y, p), v in cum.items() if p == 4}

    out: dict[pd.Timestamp, float] = {}
    for r in df.itertuples():
        y, p = int(r.year), int(r.period)
        if p == 4:
            out[r.report_date] = cum.get((y, 4), np.nan)
            continue
        prev_annual = annual.get(y - 1)
        prev_cum = cum.get((y - 1, p))
        this_cum = cum.get((y, p))
        out[r.report_date] = (
            prev_annual - prev_cum + this_cum
            if None not in (prev_annual, prev_cum, this_cum)
            else np.nan
        )
    return pd.Series(out, name=f"{col}_ttm").sort_index()


def build_multiples(
    prices: pd.Series,
    fin: pd.DataFrame,
    *,
    profit_col: str = "net_profit",
) -> pd.DataFrame:
    """逐日历史估值倍数（PIT 对齐）。

    prices 必须是**不复权**收盘价 —— 市值 = 股价 × 股本，用复权价会算出错误的市值。
    每个交易日只使用当日已公告的财报，避免前视偏差。
    """
    if fin.empty or prices.empty:
        return pd.DataFrame()

    ttm = ttm_from_cumulative(fin, profit_col)
    f = fin.copy()
    f["report_date"] = pd.to_datetime(f["report_date"])
    f["notice_date"] = pd.to_datetime(f["notice_date"])
    f["ttm_profit"] = f["report_date"].map(ttm)

    # 对每个交易日取当时已公告的最新一期
    f = f.sort_values("notice_date")
    cols = ["notice_date", "ttm_profit", "bps", "total_share", "report_name", "report_date"]
    known = f[[c for c in cols if c in f]].dropna(subset=["notice_date"])

    px = prices.sort_index()
    left = pd.DataFrame({"date": px.index, "close": px.values})
    right = known.rename(columns={"notice_date": "date"})
    # parquet 往返会产生 ms/us 精度差异，merge_asof 要求连接键 dtype 完全一致
    for frame in (left, right):
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")

    merged = pd.merge_asof(
        left.sort_values("date"), right.sort_values("date"),
        on="date", direction="backward",
    ).set_index("date")

    merged["market_cap"] = merged["close"] * merged["total_share"]
    # 盈利趋近于零时 PE 无意义，直接置 NaN 而非输出上万倍的数字
    profit = merged["ttm_profit"]
    merged["pe_ttm"] = (merged["market_cap"] / profit.where(profit > 0))
    merged["pb"] = merged["close"] / merged["bps"].where(merged["bps"] > 0)
    merged["roe_ttm"] = (profit / (merged["bps"] * merged["total_share"])
                         .where(merged["bps"] > 0)) * 100
    return merged


def percentile_of(series: pd.Series, value: float) -> float | None:
    """当前值在历史序列中的分位（0~1）。"""
    s = series.dropna()
    if s.empty or value is None or not np.isfinite(value):
        return None
    return float((s <= value).sum() / len(s))


def justified_pb(roe: float, cost_of_equity: float, growth: float) -> float:
    """剩余收益模型下的合理 PB：PB = (ROE - g) / (k - g)。

    要求 k > g，否则模型发散（隐含永续增长快于折现率，无经济意义）。
    """
    if cost_of_equity <= growth:
        raise ValueError(f"要求 k > g，当前 k={cost_of_equity:.3f} g={growth:.3f}")
    return (roe - growth) / (cost_of_equity - growth)


def implied_roe(pb: float, cost_of_equity: float, growth: float) -> float:
    """由当前 PB 反推市场隐含的永续 ROE。

    这是本模块最有用的一个数：它把「贵不贵」变成一个可证伪的问题 ——
    只需对照公司历史上是否达到过这个 ROE 水平。
    """
    if cost_of_equity <= growth:
        raise ValueError(f"要求 k > g，当前 k={cost_of_equity:.3f} g={growth:.3f}")
    return pb * (cost_of_equity - growth) + growth


@dataclass
class ValuationInputs:
    code: str
    name: str
    price: float
    total_share: float
    bps: float
    ttm_profit: float
    report_name: str
    notice_date: pd.Timestamp

    @property
    def market_cap(self) -> float:
        return self.price * self.total_share

    @property
    def equity(self) -> float:
        return self.bps * self.total_share

    @property
    def pe_ttm(self) -> float | None:
        return self.market_cap / self.ttm_profit if self.ttm_profit > 0 else None

    @property
    def pb(self) -> float:
        return self.price / self.bps

    @property
    def roe_ttm(self) -> float:
        return self.ttm_profit / self.equity * 100


def gather_inputs(
    code: str, fin: pd.DataFrame, price: float, as_of: str | pd.Timestamp
) -> ValuationInputs:
    """汇总估值所需输入，全部取自 as_of 时点已公告的数据。"""
    snap = as_of_snapshot(fin, as_of)
    if snap.empty:
        raise ValueError(f"{code} 在 {as_of} 无已公告财报")
    row = snap.iloc[0]
    ttm = ttm_from_cumulative(fin, "net_profit")
    val = ttm.get(pd.Timestamp(row["report_date"]), np.nan)
    if not np.isfinite(val):
        raise ValueError(f"{code} {row.get('report_name')} 无法还原 TTM（缺上年同期或上年年报）")
    return ValuationInputs(
        code=code,
        name=str(row.get("name", code)),
        price=float(price),
        total_share=float(row["total_share"]),
        bps=float(row["bps"]),
        ttm_profit=float(val),
        report_name=str(row.get("report_name", "")),
        notice_date=pd.Timestamp(row["notice_date"]),
    )


def peer_comparison(
    codes: list[str],
    financials: dict[str, pd.DataFrame],
    prices: dict[str, pd.Series],
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """同业横向对比。

    取数失败或无法还原 TTM 的公司**保留在表中并标注原因**，不静默剔除 ——
    剔除后剩下的往往正是数据好看的那些，会系统性美化对比结论。
    """
    rows = []
    for code in codes:
        fin = financials.get(code)
        px = prices.get(code)
        if fin is None or fin.empty or px is None or px.empty:
            rows.append({"code": code, "备注": "无数据"})
            continue
        px = px[px.index <= pd.Timestamp(as_of)]
        if px.empty:
            rows.append({"code": code, "备注": "该时点无行情"})
            continue
        try:
            v = gather_inputs(code, fin, float(px.iloc[-1]), as_of)
        except ValueError as exc:
            rows.append({"code": code, "备注": str(exc)[:40]})
            continue

        snap = as_of_snapshot(fin, as_of).iloc[0]
        rows.append({
            "code": code,
            "name": v.name,
            "价格": v.price,
            "市值(亿)": v.market_cap / 1e8,
            "PE(TTM)": v.pe_ttm,
            "PB": v.pb,
            "ROE(TTM)%": v.roe_ttm,
            "毛利率%": snap.get("gross_margin"),
            "净利率%": snap.get("net_margin"),
            "营收同比%": snap.get("revenue_yoy"),
            "负债率%": snap.get("debt_ratio"),
            "报告期": snap.get("report_name"),
            "备注": "",
        })
    return pd.DataFrame(rows)


def pb_roe_cross_section(peers: pd.DataFrame, target: str) -> dict:
    """PB-ROE 横截面回归。

    单阶段剩余收益模型会给出「整个行业都高估」这类结论，此时无法区分
    「个股错价」与「行业整体溢价」。横截面回归把行业溢价吸收进截距，
    剩下的残差才是个股相对同业的偏离。

    样本量小时回归极不稳健，因此同时返回 n 与 R²，由调用方判断可信度。
    """
    df = peers.dropna(subset=["PB", "ROE(TTM)%"]).copy()
    df = df[df["ROE(TTM)%"] > 0]        # ROE 为负时 PB-ROE 关系不成立
    n = len(df)
    if n < 4:
        return {"n": n, "usable": False, "reason": f"有效样本仅 {n} 家，不足以回归"}

    x = df["ROE(TTM)%"].to_numpy(float)
    y = df["PB"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    df["拟合PB"] = pred
    df["残差"] = y - pred
    df["相对偏离"] = df["残差"] / df["拟合PB"]

    base = {"n": n, "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2), "table": df}

    # 斜率为负意味着「ROE 越高、估值越低」，与剩余收益模型的方向相反。
    # 此时回归不是在刻画估值规律，而是在拟合噪声或被单个异常值主导，
    # 由它推出的「低估/高估」纯属数字游戏，必须拒绝输出。
    if slope <= 0:
        return {**base, "usable": False,
                "reason": (f"回归斜率 {slope:.3f} 为负，与「ROE 越高 PB 越高」的理论方向相悖，"
                           f"R²={r2:.3f}。该样本不存在可用的 PB-ROE 关系，拒绝据此判断相对估值")}
    if r2 < 0.3:
        return {**base, "usable": False,
                "reason": f"R²={r2:.3f}，ROE 对 PB 几无解释力，回归结果不可用"}

    row = df[df["code"] == target]
    return {
        **base, "usable": True,
        "target_residual": float(row["残差"].iloc[0]) if not row.empty else None,
        "target_deviation": float(row["相对偏离"].iloc[0]) if not row.empty else None,
        "target_fitted_pb": float(row["拟合PB"].iloc[0]) if not row.empty else None,
    }


def pb_to_roe_ratio(peers: pd.DataFrame, target: str) -> pd.DataFrame:
    """PB / ROE 比值对比。

    横截面回归在小样本下不稳健时的替代方案：直接比「每单位 ROE 付出多少 PB」，
    不依赖线性假设，也不会被单个异常值改变全局斜率。数值越低越便宜。
    """
    df = peers.dropna(subset=["PB", "ROE(TTM)%"]).copy()
    df = df[df["ROE(TTM)%"] > 0]
    if df.empty:
        return df
    df["PB/ROE"] = df["PB"] / df["ROE(TTM)%"]
    med = df["PB/ROE"].median()
    df["相对中位数"] = df["PB/ROE"] / med - 1
    df["_is_target"] = df["code"] == target
    return df.sort_values("PB/ROE")


@dataclass
class Scenario:
    """一个估值情景。每个字段都是**假设**，必须在报告中显式列出。"""

    name: str
    roe: float                  # 稳态 ROE (%)
    cost_of_equity: float       # 折现率 (%)
    growth: float               # 永续增长 (%)
    note: str = ""

    def target_pb(self) -> float:
        return justified_pb(self.roe, self.cost_of_equity, self.growth)

    def target_price(self, bps: float) -> float:
        return self.target_pb() * bps


@dataclass
class ValuationResult:
    inputs: ValuationInputs
    scenarios: list[Scenario]
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    peers: pd.DataFrame = field(default_factory=pd.DataFrame)

    def scenario_table(self) -> pd.DataFrame:
        rows = []
        for s in self.scenarios:
            tp = s.target_price(self.inputs.bps)
            rows.append({
                "情景": s.name,
                "稳态ROE%": s.roe,
                "折现率%": s.cost_of_equity,
                "永续增长%": s.growth,
                "合理PB": s.target_pb(),
                "目标价": tp,
                "相对现价": tp / self.inputs.price - 1,
                "说明": s.note,
            })
        return pd.DataFrame(rows)

    def sensitivity(
        self, roes: list[float], ks: list[float], growth: float = 3.0
    ) -> pd.DataFrame:
        """目标价对 ROE 与折现率的敏感性。单一目标价没有意义，区间才有。"""
        data = {}
        for k in ks:
            data[f"k={k:.0f}%"] = [
                justified_pb(r, k, growth) * self.inputs.bps for r in roes
            ]
        return pd.DataFrame(data, index=[f"ROE={r:.0f}%" for r in roes])
