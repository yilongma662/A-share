"""估值模块测试。

重点：TTM 还原（A 股累计口径极易算错）与「拒绝输出不可信结论」的守卫。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gvs.research.valuation import (
    fair_value_range,
    normalized_roe,
    gather_inputs,
    implied_roe,
    justified_pb,
    pb_roe_cross_section,
    pb_to_roe_ratio,
    percentile_of,
    ttm_from_cumulative,
)


@pytest.fixture
def cumulative() -> pd.DataFrame:
    """A 股累计口径财报。数值取自 002185 实测，便于与东财官方结果对照。"""
    rows = [
        ("2025一季报", "2025-03-31", "2025-04-30", -18528648.84),
        ("2025中报", "2025-06-30", "2025-08-19", 226478541.12),
        ("2025三季报", "2025-09-30", "2025-10-28", 542637017.10),
        ("2025年报", "2025-12-31", "2026-03-31", 710508576.43),
        ("2026一季报", "2026-03-31", "2026-04-29", 86786407.33),
    ]
    return pd.DataFrame(
        [{"report_name": a, "report_date": b, "notice_date": c, "net_profit": d,
          "bps": 5.4846, "total_share": 3323423616.0, "code": "002185",
          "name": "华天科技"} for a, b, c, d in rows]
    )


def test_年报TTM即为年度累计(cumulative):
    ttm = ttm_from_cumulative(cumulative)
    assert ttm[pd.Timestamp("2025-12-31")] == pytest.approx(710508576.43)


def test_季报TTM由累计口径还原(cumulative):
    """TTM(2026Q1) = 年报2025 - 一季报2025 + 一季报2026。

    直接把季报当 TTM，或简单乘 4，都会得到完全不同的数字。
    """
    ttm = ttm_from_cumulative(cumulative)
    expected = 710508576.43 - (-18528648.84) + 86786407.33
    assert ttm[pd.Timestamp("2026-03-31")] == pytest.approx(expected)
    assert ttm[pd.Timestamp("2026-03-31")] / 1e8 == pytest.approx(8.158, abs=0.001)


def test_缺上年数据时TTM为NaN(cumulative):
    """无法还原时必须是 NaN，不得用近似值顶替。"""
    only_first = cumulative[cumulative["report_date"] == "2025-03-31"]
    ttm = ttm_from_cumulative(only_first)
    assert pd.isna(ttm.iloc[0])


def test_PE与东财官方口径一致(cumulative):
    """独立算出的 PE(TTM) 应与东财 stock/get 接口 f164=63.02 吻合。"""
    v = gather_inputs("002185", cumulative, price=15.47, as_of="2026-07-31")
    assert v.pe_ttm == pytest.approx(63.02, abs=0.05)
    assert v.market_cap / 1e8 == pytest.approx(514.13, abs=0.05)
    assert v.roe_ttm == pytest.approx(4.48, abs=0.02)


def test_ROE等于折现率时合理PB为1():
    """剩余收益模型的基准情形：不创造也不毁灭价值时 PB = 1。"""
    assert justified_pb(roe=10.0, cost_of_equity=10.0, growth=3.0) == pytest.approx(1.0)


def test_ROE低于折现率时合理PB小于1():
    assert justified_pb(roe=4.5, cost_of_equity=10.0, growth=3.0) < 1.0


def test_隐含ROE与合理PB互为逆运算():
    pb = justified_pb(roe=12.0, cost_of_equity=10.0, growth=3.0)
    assert implied_roe(pb, cost_of_equity=10.0, growth=3.0) == pytest.approx(12.0)


def test_增长率不低于折现率时模型发散():
    with pytest.raises(ValueError, match="k > g"):
        justified_pb(roe=10.0, cost_of_equity=3.0, growth=3.0)


def test_回归斜率为负时拒绝给出结论():
    """ROE 越高 PB 反而越低，与理论相悖，此时残差没有估值含义。"""
    peers = pd.DataFrame({
        "code": ["A", "B", "C", "D", "E"],
        "PB": [9.0, 6.0, 4.0, 3.0, 2.0],
        "ROE(TTM)%": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    cs = pb_roe_cross_section(peers, "A")
    assert cs["usable"] is False
    assert cs["slope"] < 0
    assert "相悖" in cs["reason"]


def test_解释力不足时拒绝给出结论():
    rng = np.random.default_rng(3)
    peers = pd.DataFrame({
        "code": [f"S{i}" for i in range(12)],
        "ROE(TTM)%": rng.uniform(2, 12, 12),
        "PB": rng.uniform(1, 9, 12),
    })
    cs = pb_roe_cross_section(peers, "S0")
    if cs.get("slope", 0) > 0:          # 斜率为正时才轮到 R² 把关
        assert cs["usable"] is False and "解释力" in cs["reason"]


def test_样本过少时拒绝回归():
    peers = pd.DataFrame({"code": ["A", "B"], "PB": [2.0, 3.0], "ROE(TTM)%": [5.0, 8.0]})
    assert pb_roe_cross_section(peers, "A")["usable"] is False


def test_ROE为负的公司不进入横截面():
    peers = pd.DataFrame({
        "code": ["A", "B", "C", "D", "E"],
        "PB": [2.0, 3.0, 4.0, 5.0, 2.8],
        "ROE(TTM)%": [4.0, 6.0, 8.0, 10.0, -1.0],
    })
    assert pb_roe_cross_section(peers, "A")["n"] == 4


def test_PB_ROE比值排序():
    peers = pd.DataFrame({
        "code": ["贵", "便宜"], "name": ["贵", "便宜"],
        "PB": [9.0, 2.0], "ROE(TTM)%": [3.0, 8.0],
    })
    r = pb_to_roe_ratio(peers, "便宜")
    assert r.iloc[0]["code"] == "便宜", "PB/ROE 最低者应排在最前"


def test_分位数计算():
    s = pd.Series(range(101), dtype=float)
    assert percentile_of(s, 50.0) == pytest.approx(0.5049, abs=0.001)
    assert percentile_of(s, 0.0) == pytest.approx(1 / 101, abs=0.001)


def test_中周期ROE只取年报():
    """季报 ROE 未年化，混入统计会把周期中枢严重拉低。"""
    rows = []
    for y in range(2014, 2026):
        rows.append({"report_date": f"{y}-12-31", "roe": 8.0})
        rows.append({"report_date": f"{y}-03-31", "roe": 0.5})   # 季报，应被排除
    norm = normalized_roe(pd.DataFrame(rows))
    assert norm["usable"] is True
    assert norm["n"] == 12
    assert norm["mean"] == pytest.approx(8.0), "混入季报会把均值拉低到 4.25 附近"


def test_年报样本不足时拒绝正常化():
    rows = [{"report_date": f"{y}-12-31", "roe": 8.0} for y in (2023, 2024, 2025)]
    assert normalized_roe(pd.DataFrame(rows))["usable"] is False


def test_合理价区间同时给出基本面与同业锚定(cumulative):
    v = gather_inputs("002185", cumulative, price=15.47, as_of="2026-07-31")
    fin = pd.concat([
        cumulative,
        pd.DataFrame([{"report_name": f"{y}年报", "report_date": f"{y}-12-31",
                       "notice_date": f"{y + 1}-03-31", "net_profit": 5e8,
                       "bps": 5.0, "total_share": 3.3e9, "roe": r, "code": "002185"}
                      for y, r in zip(range(2014, 2025),
                                      [14.7, 11.8, 8.2, 9.7, 7.1, 4.3, 8.7, 14.0, 4.9, 1.4, 3.8])]),
    ], ignore_index=True)
    peers = pd.DataFrame({
        "code": ["002185", "A", "B", "C"], "name": ["华天", "A", "B", "C"],
        "PB": [2.82, 4.06, 5.50, 4.09], "ROE(TTM)%": [4.48, 7.93, 9.24, 5.74],
    })
    fv = fair_value_range(v, fin, peers)
    methods = " ".join(fv["方法"])
    assert "剩余收益" in methods and "同业PB/ROE" in methods
    assert (fv["合理价"] > 0).all()
    # 强周期股的两类锚定必然分歧，区间不应被压缩成一个点
    assert fv["合理价"].max() / fv["合理价"].min() > 1.5
