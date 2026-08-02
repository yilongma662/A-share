"""回测引擎测试。

回测代码的错误不会崩溃，只会产出好看的假结果，因此每条防护都必须有测试。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gvs.backtest.engine import rebalance_dates, run_backtest
from gvs.backtest.metrics import evaluate, max_drawdown, quantile_monotonicity
from gvs.config import BacktestConfig, TradingCost


@pytest.fixture
def prices() -> pd.DataFrame:
    """三只股票 500 个交易日。A 稳定上涨，B 稳定下跌，C 含停牌缺口。"""
    idx = pd.bdate_range("2024-01-01", periods=500)
    n = len(idx)
    a = pd.Series(10 * np.cumprod(1 + np.full(n, 0.0008)), index=idx)
    b = pd.Series(10 * np.cumprod(1 + np.full(n, -0.0005)), index=idx)
    c = pd.Series(10 * np.cumprod(1 + np.full(n, 0.0002)), index=idx)
    c.iloc[100:120] = np.nan          # 停牌
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_只买上涨股应跑赢只买下跌股(prices):
    up = run_backtest(prices, lambda d, avail: ["A"],
                      BacktestConfig(start="2024-01-01", top_n=1))
    down = run_backtest(prices, lambda d, avail: ["B"],
                        BacktestConfig(start="2024-01-01", top_n=1))
    assert up.performance.annual_return > down.performance.annual_return
    assert up.performance.total_return > 0 > down.performance.total_return


def test_交易成本降低收益(prices):
    free = BacktestConfig(start="2024-01-01", cost=TradingCost(
        commission_rate=0, min_commission=0, stamp_duty_rate=0,
        transfer_fee_rate=0, slippage_rate=0))
    costly = BacktestConfig(start="2024-01-01", cost=TradingCost(slippage_rate=0.005))

    def alternate(d, avail):
        return ["A"] if d.month % 2 == 0 else ["B"]

    r_free = run_backtest(prices, alternate, free)
    r_cost = run_backtest(prices, alternate, costly)
    assert r_cost.performance.total_return < r_free.performance.total_return, \
        "加入交易成本后收益必须下降，否则说明成本未被计入"


def test_停牌股不被选中(prices):
    """停牌期间价格为 NaN，selector 收到的 available 不应包含该股票。"""
    seen: list[list[str]] = []

    def record(d, avail):
        seen.append(list(avail))
        return ["A"]

    run_backtest(prices, record, BacktestConfig(start="2024-01-01"))
    halted = [s for s in seen if "C" not in s]
    assert halted, "测试数据含停牌区间，应至少有一期 C 不可交易"


def test_选股为空时保留原持仓并告警(prices):
    def sometimes_empty(d, avail):
        return [] if d.month == 6 else ["A"]

    result = run_backtest(prices, sometimes_empty, BacktestConfig(start="2024-01-01"))
    assert any("选股结果为空" in w for w in result.warnings)


def test_换手率计算(prices):
    """完全换仓的单边换手率为 1。"""
    def flip(d, avail):
        return ["A"] if d.month % 2 == 0 else ["B"]

    r = run_backtest(prices, flip, BacktestConfig(start="2024-01-01"))
    assert r.turnover.max() == pytest.approx(1.0, abs=0.01)


def test_调仓日为周期末交易日():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    dates = rebalance_dates(idx, "M")
    assert len(dates) == 3
    assert dates[0] == pd.Timestamp("2024-01-31")


def test_最大回撤():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1], index=pd.to_datetime(
        ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]))
    mdd, _ = max_drawdown(eq)
    assert mdd == pytest.approx(-0.25)


def test_绩效指标(prices):
    rets = prices["A"].pct_change().dropna()
    perf = evaluate(rets)
    assert perf.annual_return > 0
    assert perf.max_drawdown <= 0
    assert 0 <= perf.win_rate <= 1


def test_分组单调性():
    """构造分数与收益完全正相关的数据，Q5 均值必须高于 Q1。"""
    rng = np.random.default_rng(42)
    scores = pd.Series(rng.normal(size=200))
    fwd = scores * 0.05 + rng.normal(scale=0.01, size=200)
    table = quantile_monotonicity(scores, fwd, n_groups=5)
    assert table.loc["Q5", "mean"] > table.loc["Q1", "mean"]


def test_样本不足时拒绝分组():
    with pytest.raises(ValueError, match="样本量"):
        quantile_monotonicity(pd.Series([1.0, 2.0]), pd.Series([0.1, 0.2]), n_groups=5)
