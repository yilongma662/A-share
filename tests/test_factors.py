"""因子计算测试。

重点在缺失数据的处理 —— 缺失被静默当作"中性"是打分体系最隐蔽的失真来源。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gvs.factors.growth import FactorResult, composite_score, peg, revenue_growth


@pytest.fixture
def snap() -> pd.DataFrame:
    """20 只股票的 PIT 快照。估值字段（pe_ttm）整体缺失，模拟财务表不含估值的真实情况。"""
    rng = np.random.default_rng(7)
    n = 20
    return pd.DataFrame({
        "code": [f"{i:06d}" for i in range(n)],
        "report_name": ["2026一季报"] * n,
        "revenue_yoy": rng.normal(20, 15, n),
        "net_profit_yoy": rng.normal(25, 30, n),
        "net_profit_deducted_yoy": rng.normal(22, 28, n),
        "roe": rng.normal(8, 4, n),
        "gross_margin": rng.normal(25, 10, n),
        "net_margin": rng.normal(8, 5, n),
        "ocf_per_share": rng.normal(0.5, 0.3, n),
        "eps": rng.normal(0.4, 0.2, n),
        "debt_ratio": rng.normal(45, 15, n),
        "revenue": rng.normal(5e9, 1e9, n),
    })


def test_覆盖率计算():
    r = FactorResult("t", pd.Series([1.0, 2.0, np.nan, np.nan]), "测试")
    assert r.coverage == 0.5


def test_方向为负的因子排序反转():
    """direction=-1 表示越小越好，zscore 后应变为越大越好。"""
    low_is_good = FactorResult("t", pd.Series([1.0, 2.0, 3.0]), "测试", direction=-1)
    z = low_is_good.zscore()
    assert z.iloc[0] > z.iloc[-1]


def test_去极值防止单只股票主导():
    """未去极值时，一个 1000 倍的异常值会把其余样本压成同一个数。"""
    normal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
    with_outlier = pd.concat([normal, pd.Series([10000.0])], ignore_index=True)
    z = FactorResult("t", with_outlier, "测试").zscore(winsor=0.05)
    assert z.iloc[:20].std() > 0.5, "去极值后正常样本之间仍应有区分度"


def test_增速为负时PEG无意义():
    df = pd.DataFrame({"pe_ttm": [20.0, 30.0], "net_profit_yoy": [10.0, -5.0]})
    v = peg(df).values
    assert v.iloc[0] == pytest.approx(2.0)
    assert pd.isna(v.iloc[1]), "增速为负时 PEG 无经济含义，必须为 NaN 而非负值"


def test_整块因子缺失时重新归一化权重(snap):
    """估值块完全缺失时，剩余权重必须归一化到 1。

    否则 fillna(0) 会把缺失块当作中性分计入，权重和不为 1，
    各维度相对影响被悄悄改变 —— 分数看起来正常，实际已失真。
    """
    scored = composite_score(snap)
    assert scored.attrs["empty_blocks"] == ["value"]
    eff = scored.attrs["effective_weights"]
    assert "value" not in eff
    assert sum(eff.values()) == pytest.approx(1.0)
    # 名义 0.45:0.35 归一化后应保持相对比例
    assert eff["growth"] / eff["quality"] == pytest.approx(0.45 / 0.35)


def test_剔除低覆盖率因子并记录(snap):
    scored = composite_score(snap)
    dropped = scored.attrs["dropped_factors"]
    assert any("pe_ttm" in d for d in dropped)
    assert any("覆盖率" in d for d in dropped), "剔除原因必须可追溯"


def test_全部因子缺失时拒绝打分():
    empty = pd.DataFrame({"code": ["000001", "000002"]})
    with pytest.raises(ValueError, match="无法打分"):
        composite_score(empty)


def test_打分按降序排列(snap):
    scored = composite_score(snap)
    s = scored["score_total"].dropna()
    assert (s.diff().dropna() <= 1e-9).all()


def test_缺失全部维度的个股不给分(snap):
    """无数据的个股必须为 NaN，不能因 fillna(0) 而获得中性分排在中间。"""
    snap2 = pd.concat([snap, pd.DataFrame([{"code": "999999"}])], ignore_index=True)
    scored = composite_score(snap2)
    assert pd.isna(scored.set_index("code").loc["999999", "score_total"])


def test_营收增速因子直接取用同比字段(snap):
    r = revenue_growth(snap)
    assert r.direction == 1
    pd.testing.assert_series_equal(
        r.values.reset_index(drop=True),
        snap["revenue_yoy"].reset_index(drop=True),
        check_names=False,
    )
