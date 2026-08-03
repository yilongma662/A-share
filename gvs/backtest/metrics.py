"""绩效指标。

宪章第二条：只报绝对收益是没有意义的，必须同时给出基准对比、回撤与换手。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 243  # A 股年均交易日


@dataclass
class Performance:
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    max_drawdown_days: int
    calmar: float
    win_rate: float
    periods: int
    benchmark_return: float | None = None
    excess_return: float | None = None
    information_ratio: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"区间总收益      {self.total_return:>8.2%}",
            f"年化收益        {self.annual_return:>8.2%}",
            f"年化波动        {self.annual_volatility:>8.2%}",
            f"夏普比率        {self.sharpe:>8.2f}",
            f"最大回撤        {self.max_drawdown:>8.2%}  (持续 {self.max_drawdown_days} 天)",
            f"Calmar          {self.calmar:>8.2f}",
            f"胜率            {self.win_rate:>8.2%}",
        ]
        if self.benchmark_return is not None:
            lines += [
                f"基准收益        {self.benchmark_return:>8.2%}",
                f"超额收益        {self.excess_return:>8.2%}",
                f"信息比率        {self.information_ratio:>8.2f}",
            ]
        return "\n".join(lines)


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """最大回撤及其持续天数。"""
    if equity.empty:
        return 0.0, 0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    mdd = float(dd.min())
    peak_idx = equity.loc[:trough].idxmax()
    days = int((trough - peak_idx).days) if hasattr(trough, "days") or \
        isinstance(trough, pd.Timestamp) else int(equity.index.get_loc(trough) -
                                                  equity.index.get_loc(peak_idx))
    return mdd, days


def evaluate(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.015,
    freq: int = TRADING_DAYS,
) -> Performance:
    """returns 为周期收益率序列（非累计）。"""
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("收益序列为空，无法评估")

    equity = (1 + r).cumprod()
    years = len(r) / freq
    total = float(equity.iloc[-1] - 1)
    annual = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    vol = float(r.std(ddof=0) * np.sqrt(freq))
    sharpe = float((annual - risk_free) / vol) if vol > 0 else 0.0
    mdd, mdd_days = max_drawdown(equity)
    calmar = float(annual / abs(mdd)) if mdd < 0 else 0.0
    win = float((r > 0).sum() / len(r))

    perf = Performance(
        total_return=total, annual_return=annual, annual_volatility=vol,
        sharpe=sharpe, max_drawdown=mdd, max_drawdown_days=mdd_days,
        calmar=calmar, win_rate=win, periods=len(r),
    )

    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna().astype(float)
        if not b.empty:
            aligned = r.reindex(b.index)
            b_equity = (1 + b).cumprod()
            b_years = len(b) / freq
            perf.benchmark_return = float(b_equity.iloc[-1] - 1)
            b_annual = float(b_equity.iloc[-1] ** (1 / b_years) - 1) if b_years > 0 else 0.0
            perf.excess_return = annual - b_annual
            active = aligned - b
            te = float(active.std(ddof=0) * np.sqrt(freq))
            perf.information_ratio = float(active.mean() * freq / te) if te > 0 else 0.0
    return perf


def quantile_monotonicity(
    scores: pd.Series, forward_returns: pd.Series, n_groups: int = 5
) -> pd.DataFrame:
    """分组单调性检验。

    这是判断因子是否真实有效的核心证据：若 Q1..Q5 收益不单调，
    说明因子与收益无稳定关系，即便多空收益为正也可能只是噪声。
    """
    df = pd.DataFrame({"score": scores, "fwd": forward_returns}).dropna()
    if len(df) < n_groups * 2:
        raise ValueError(f"样本量 {len(df)} 不足以分 {n_groups} 组")
    df["group"] = pd.qcut(df["score"].rank(method="first"), n_groups,
                          labels=[f"Q{i+1}" for i in range(n_groups)])
    out = df.groupby("group", observed=True)["fwd"].agg(["mean", "std", "count"])
    out["cumulative_rank"] = out["mean"].rank()
    return out
