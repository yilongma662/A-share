"""Point-in-time 对齐测试。

这是全项目最重要的测试 —— 前视偏差不会让程序崩溃，只会让回测结果变好看。
"""
from __future__ import annotations

import pandas as pd
import pytest

from gvs.factors.pit import (
    LookaheadError,
    align_price_to_signal,
    as_of_snapshot,
    assert_no_lookahead,
    build_pit_panel,
)


@pytest.fixture
def financials() -> pd.DataFrame:
    """两只股票的财报，公告日均滞后于报告期。数据结构取自东财实测。"""
    return pd.DataFrame([
        {"code": "002185", "report_date": "2025-12-31", "notice_date": "2026-03-31",
         "revenue_yoy": 19.03},
        {"code": "002185", "report_date": "2026-03-31", "notice_date": "2026-04-29",
         "revenue_yoy": 34.49},
        {"code": "600519", "report_date": "2025-12-31", "notice_date": "2026-04-02",
         "revenue_yoy": 8.10},
    ])


def test_未公告的财报不可见(financials):
    """报告期已过但尚未公告时，该期数据必须不可见 —— 这是最核心的约束。"""
    snap = as_of_snapshot(financials, "2026-04-10")
    row = snap[snap["code"] == "002185"].iloc[0]
    assert pd.Timestamp(row["report_date"]) == pd.Timestamp("2025-12-31"), \
        "2026-04-10 时一季报（公告日 04-29）尚未披露，不应被选中"
    assert row["revenue_yoy"] == 19.03


def test_公告后财报可见(financials):
    snap = as_of_snapshot(financials, "2026-05-01")
    row = snap[snap["code"] == "002185"].iloc[0]
    assert pd.Timestamp(row["report_date"]) == pd.Timestamp("2026-03-31")
    assert row["revenue_yoy"] == 34.49


def test_公告日当天即可见(financials):
    """公告日当天收盘后即可使用，边界取闭区间。"""
    snap = as_of_snapshot(financials, "2026-04-29")
    assert pd.Timestamp(snap[snap["code"] == "002185"].iloc[0]["report_date"]) == \
        pd.Timestamp("2026-03-31")


def test_同一报告期取最后公告版本():
    """财报追溯调整：同一报告期有多个版本时取 as_of 前最后一次公告的。"""
    df = pd.DataFrame([
        {"code": "000001", "report_date": "2025-12-31", "notice_date": "2026-03-01",
         "revenue_yoy": 10.0},
        {"code": "000001", "report_date": "2025-12-31", "notice_date": "2026-06-01",
         "revenue_yoy": 8.5},
    ])
    assert as_of_snapshot(df, "2026-04-01").iloc[0]["revenue_yoy"] == 10.0
    assert as_of_snapshot(df, "2026-07-01").iloc[0]["revenue_yoy"] == 8.5


def test_早于所有公告日返回空(financials):
    assert as_of_snapshot(financials, "2020-01-01").empty


def test_记录数据滞后天数(financials):
    snap = as_of_snapshot(financials, "2026-05-29")
    assert snap[snap["code"] == "002185"].iloc[0]["_data_lag_days"] == 30


def test_面板不含前视数据(financials):
    panel = build_pit_panel(financials, ["2026-04-10", "2026-05-01", "2026-06-01"])
    assert_no_lookahead(panel)


def test_前视检测能抓出违规():
    bad = pd.DataFrame([{"notice_date": "2026-04-29", "_as_of": "2026-04-10"}])
    with pytest.raises(LookaheadError, match="前视数据"):
        assert_no_lookahead(bad)


def test_T1成交():
    """信号在收盘产生，最早次日成交。"""
    bars = pd.DataFrame({"date": pd.to_datetime(
        ["2026-07-29", "2026-07-30", "2026-07-31"])})
    assert align_price_to_signal(pd.Timestamp("2026-07-29"), bars, delay=1) == \
        pd.Timestamp("2026-07-30")


def test_最后一日无次日则无法成交():
    bars = pd.DataFrame({"date": pd.to_datetime(["2026-07-30", "2026-07-31"])})
    assert align_price_to_signal(pd.Timestamp("2026-07-31"), bars, delay=1) is None
