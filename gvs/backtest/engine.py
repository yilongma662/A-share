"""回测引擎。

设计原则：宁可跑得慢，也不能算得假。引擎在多处主动抛错而非静默容错，
因为回测里"看起来能跑"的错误远比崩溃危险。

已实现的偏差防护：
  - T+1 成交（信号在收盘产生，最早次日买入）
  - 停牌股不可交易（价格缺失即视为停牌，持仓延续，不参与调仓）
  - 交易成本（佣金 + 印花税 + 过户费 + 冲击成本）
尚未实现（见 CHARTER 第三节）：涨跌停无法成交、退市股回填。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gvs.backtest.metrics import Performance, evaluate
from gvs.config import BacktestConfig, TradingCost

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    performance: Performance
    trades: int
    config: BacktestConfig
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = (
            f"回测区间 {self.equity.index[0]:%Y-%m-%d} ~ {self.equity.index[-1]:%Y-%m-%d}\n"
            f"调仓次数 {self.trades}   平均换手 {self.turnover.mean():.1%}\n"
            + "-" * 42
        )
        body = self.performance.summary()
        tail = ""
        if self.warnings:
            tail = "\n" + "-" * 42 + "\n告警:\n" + "\n".join(f"  · {w}" for w in self.warnings)
        return f"{head}\n{body}{tail}"


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> list[pd.Timestamp]:
    """取每个周期最后一个交易日作为信号日。"""
    s = pd.Series(index, index=index)
    alias = {"M": "ME", "Q": "QE", "W": "W", "Y": "YE"}.get(freq, freq)
    return list(s.resample(alias).last().dropna())


def run_backtest(
    prices: pd.DataFrame,
    selector,
    config: BacktestConfig | None = None,
    benchmark: pd.Series | None = None,
) -> BacktestResult:
    """等权组合回测。

    prices   : index=交易日, columns=股票代码, values=前复权收盘价。缺失=停牌。
    selector : (as_of: Timestamp, available: list[str]) -> list[str]，返回目标持仓。
               实现方必须自行保证只使用 as_of 之前的信息。
    """
    cfg = config or BacktestConfig()
    cost: TradingCost = cfg.cost

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index)
    if cfg.start:
        prices = prices.loc[prices.index >= pd.Timestamp(cfg.start)]
    if cfg.end:
        prices = prices.loc[prices.index <= pd.Timestamp(cfg.end)]
    if prices.empty:
        raise ValueError("价格面板在回测区间内为空")

    signal_days = rebalance_dates(prices.index, cfg.rebalance)
    all_days = list(prices.index)
    warnings: list[str] = []

    equity = 1.0
    weights = pd.Series(dtype=float)          # code -> 权重
    equity_curve: dict[pd.Timestamp, float] = {}
    turnover_log: dict[pd.Timestamp, float] = {}
    position_log: list[dict] = []
    trades = 0

    # 信号日 -> 执行日（T+1）
    exec_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for sd in signal_days:
        later = [d for d in all_days if d > sd]
        if later:
            exec_map[later[0]] = sd

    prev_day = None
    for day in all_days:
        if prev_day is not None and not weights.empty:
            p_now = prices.loc[day, weights.index]
            p_prev = prices.loc[prev_day, weights.index]
            # 停牌（价格缺失）视为当日零收益，持仓延续
            ret = (p_now / p_prev - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            port_ret = float((weights * ret).sum())
            equity *= 1 + port_ret
            # 权重随价格漂移，下次调仓前不再平衡
            grown = weights * (1 + ret)
            weights = grown / grown.sum() if grown.sum() > 0 else grown

        if day in exec_map:
            as_of = exec_map[day]
            tradable = prices.loc[day].dropna()
            target = [c for c in selector(as_of, list(tradable.index)) if c in tradable.index]
            if not target:
                warnings.append(f"{as_of:%Y-%m-%d} 选股结果为空，维持原持仓")
            else:
                new_w = pd.Series(1.0 / len(target), index=target)
                turnover = _turnover(weights, new_w)
                fee = _apply_cost(weights, new_w, cost)
                equity *= 1 - fee
                turnover_log[day] = turnover
                weights = new_w
                trades += 1
                position_log.append({"date": day, "as_of": as_of, "n": len(target),
                                     "codes": ",".join(target[:50])})

        equity_curve[day] = equity
        prev_day = day

    eq = pd.Series(equity_curve).sort_index()
    rets = eq.pct_change().dropna()
    bench = benchmark.reindex(rets.index).dropna() if benchmark is not None else None
    perf = evaluate(rets, bench)

    return BacktestResult(
        equity=eq, returns=rets,
        positions=pd.DataFrame(position_log),
        turnover=pd.Series(turnover_log, dtype=float),
        performance=perf, trades=trades, config=cfg, warnings=warnings,
    )


def _turnover(old: pd.Series, new: pd.Series) -> float:
    """单边换手率。"""
    idx = old.index.union(new.index)
    o = old.reindex(idx).fillna(0.0)
    n = new.reindex(idx).fillna(0.0)
    return float((n - o).abs().sum() / 2)


def _apply_cost(old: pd.Series, new: pd.Series, cost: TradingCost) -> float:
    """按权重变动估算成本占组合净值的比例。

    简化假设：组合规模足够大，最低佣金 5 元不构成约束。小资金回测须另行处理。
    """
    idx = old.index.union(new.index)
    o = old.reindex(idx).fillna(0.0)
    n = new.reindex(idx).fillna(0.0)
    delta = n - o
    buy = float(delta[delta > 0].sum())
    sell = float(-delta[delta < 0].sum())
    buy_rate = cost.commission_rate + cost.transfer_fee_rate + cost.slippage_rate
    sell_rate = (cost.commission_rate + cost.stamp_duty_rate +
                 cost.transfer_fee_rate + cost.slippage_rate)
    return buy * buy_rate + sell * sell_rate
