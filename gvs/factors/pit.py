"""Point-in-time 对齐。

宪章第三条：财务因子必须经由本模块取数。

为什么必须有这一层 —— 实测 002185 华天科技 2026 年一季报：
    报告期 REPORT_DATE = 2026-03-31
    公告日 NOTICE_DATE = 2026-04-29
两者相差 29 天。若在 2026-04-10 用报告期对齐，就等于提前 19 天读到了尚未公布的财报。
这类偏差在回测中表现为"策略非常有效"，实盘则完全失效。
"""
from __future__ import annotations

import pandas as pd


class LookaheadError(RuntimeError):
    """检测到前视偏差。这是必须中断的错误，不是警告。"""


def as_of_snapshot(
    financials: pd.DataFrame,
    as_of: str | pd.Timestamp,
    *,
    code_col: str = "code",
    notice_col: str = "notice_date",
    report_col: str = "report_date",
) -> pd.DataFrame:
    """取 as_of 时点**市场已知**的最新一期财务数据。

    只保留 notice_date <= as_of 的记录，再按 report_date 取每只股票的最新一期。
    同一报告期存在多个版本（追溯调整）时，取 as_of 前最后一次公告的版本。
    """
    if financials.empty:
        return financials

    as_of = pd.Timestamp(as_of)
    df = financials.copy()
    df[notice_col] = pd.to_datetime(df[notice_col])
    df[report_col] = pd.to_datetime(df[report_col])

    known = df[df[notice_col] <= as_of]
    if known.empty:
        return known

    # 同一 (股票, 报告期) 取公告日最晚的版本，再取每只股票报告期最新的一期
    known = known.sort_values([code_col, report_col, notice_col])
    known = known.drop_duplicates([code_col, report_col], keep="last")
    latest = known.sort_values([code_col, report_col]).drop_duplicates(code_col, keep="last")

    result = latest.reset_index(drop=True)
    result["_as_of"] = as_of
    result["_data_lag_days"] = (as_of - result[notice_col]).dt.days
    return result


def build_pit_panel(
    financials: pd.DataFrame,
    dates: list[str] | pd.DatetimeIndex,
    **kwargs,
) -> pd.DataFrame:
    """按给定调仓日序列生成 PIT 面板，供回测逐期取用。"""
    frames = [as_of_snapshot(financials, d, **kwargs) for d in pd.to_datetime(dates)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def assert_no_lookahead(
    panel: pd.DataFrame,
    *,
    as_of_col: str = "_as_of",
    notice_col: str = "notice_date",
) -> None:
    """断言面板中不存在未来数据。回测前必须调用。"""
    if panel.empty:
        return
    bad = panel[pd.to_datetime(panel[notice_col]) > pd.to_datetime(panel[as_of_col])]
    if not bad.empty:
        sample = bad[[notice_col, as_of_col]].head(3).to_dict("records")
        raise LookaheadError(
            f"检测到 {len(bad)} 条前视数据（公告日晚于评估日），样例: {sample}"
        )


def align_price_to_signal(
    signal_date: pd.Timestamp,
    bars: pd.DataFrame,
    *,
    date_col: str = "date",
    delay: int = 1,
) -> pd.Timestamp | None:
    """信号日 -> 实际成交日。

    默认 T+1 成交：信号在收盘后产生，最早只能次日买入。
    delay=0 表示当日收盘成交，仅在明确假设收盘可成交时使用，须在报告中声明。
    """
    dates = pd.to_datetime(bars[date_col]).sort_values().unique()
    future = dates[dates > pd.Timestamp(signal_date)] if delay > 0 else \
        dates[dates >= pd.Timestamp(signal_date)]
    if len(future) < delay or len(future) == 0:
        return None
    return pd.Timestamp(future[max(delay - 1, 0)])
