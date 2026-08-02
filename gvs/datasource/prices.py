"""行情服务：多源故障转移。

存在的理由 —— 实测东财 `push2his` 会对单个 IP 累积限流，触发后连续数分钟
直接断开连接（`RemoteDisconnected`），且无任何错误码提示。单源架构在这种情况下
会静默中断整个研究流程，因此行情取数必须具备自动降级能力。

**重要约束**：不同数据源的复权口径不同，混用会污染回测。
本服务在返回结果中标注 `_provider`，回测前必须检查同一标的的数据来自同一来源。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from gvs.datasource.eastmoney import DataSourceError, EastmoneyClient
from gvs.datasource.yahoo import YahooClient

log = logging.getLogger(__name__)


class AllProvidersFailed(RuntimeError):
    """所有数据源均失败。必须中断，不得当作空数据继续。"""


@dataclass
class PriceService:
    eastmoney: EastmoneyClient = field(default_factory=EastmoneyClient)
    yahoo: YahooClient = field(default_factory=YahooClient)
    allow_fallback: bool = True

    def daily_bars(self, code: str, adjust: int = 1, **kwargs) -> pd.DataFrame:
        """优先东财（历史更长、复权口径统一），失败时降级 Yahoo。"""
        errors: list[str] = []
        try:
            df = self.eastmoney.daily_bars(code, adjust=adjust, **kwargs)
            if not df.empty:
                df["_provider"] = "eastmoney"
                return df
            errors.append("eastmoney 返回空")
        except DataSourceError as exc:
            errors.append(f"eastmoney: {exc}")
            log.warning("东财取数失败，尝试降级: %s", exc)

        if not self.allow_fallback:
            raise AllProvidersFailed(f"{code} 取数失败且未启用降级: {errors}")

        try:
            df = self.yahoo.daily_bars(code)
            if not df.empty:
                # Yahoo 的 close 为不复权价，与东财 fqt=1 前复权不是同一口径
                df["_provider"] = "yahoo"
                df["adjust"] = 0
                log.warning(
                    "%s 使用 Yahoo 降级数据，复权口径为不复权（请求的是 adjust=%d），"
                    "不可与东财序列混用",
                    code, adjust,
                )
                return df
            errors.append("yahoo 返回空")
        except Exception as exc:
            errors.append(f"yahoo: {exc}")

        raise AllProvidersFailed(f"{code} 所有数据源均失败: {errors}")

    def health_check(self) -> dict[str, bool]:
        """探测各数据源可用性。限流是常态，运行长任务前应先检查。"""
        status = {}
        try:
            status["eastmoney"] = not self.eastmoney.daily_bars(
                "000001", start="20260101").empty
        except Exception:
            status["eastmoney"] = False
        try:
            status["yahoo"] = not self.yahoo.daily_bars("000001", range_="1mo").empty
        except Exception:
            status["yahoo"] = False
        return status
