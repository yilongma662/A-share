"""Yahoo Finance 行情源。

定位：东财 `push2his` 限流时的**备用源**与交叉校验源，不作为主数据源。

已知差异（使用前必须了解）：
  - 复权口径与东财不同，两者价格序列不可直接混用
  - 历史深度不如东财（东财 002185 可回溯 4507 根日线）
  - 停牌日的处理方式不同
因此策略回测应固定使用同一数据源，切换源必须重跑全部回测。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import requests

from gvs import config

log = logging.getLogger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def to_yahoo_symbol(code: str) -> str:
    """A 股代码 -> Yahoo 代码。6/9 开头为沪市 .SS，其余深市/北交所 .SZ。"""
    code = code.strip().split(".")[0]
    return f"{code}.SS" if code[0] in "69" else f"{code}.SZ"


@dataclass
class YahooClient:
    session: requests.Session | None = None
    timeout: float = config.REQUEST_TIMEOUT
    max_retries: int = 3

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update(_UA)

    def daily_bars(self, code: str, range_: str = "10y") -> pd.DataFrame:
        """日线。Yahoo 的 adjclose 为后复权口径，与东财前复权不同，两者不可混用。"""
        url = config.YAHOO_CHART_URL.format(symbol=to_yahoo_symbol(code))
        params = {"range": range_, "interval": "1d", "events": "div,split"}

        payload = None
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as exc:
                log.warning("Yahoo 请求失败 (%d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
        if payload is None:
            raise RuntimeError(f"Yahoo 获取 {code} 失败")

        results = (payload.get("chart") or {}).get("result") or []
        if not results:
            return pd.DataFrame()
        r = results[0]
        quote = (r.get("indicators") or {}).get("quote", [{}])[0]
        adj = (r.get("indicators") or {}).get("adjclose", [{}])
        adjclose = adj[0].get("adjclose") if adj else None

        df = pd.DataFrame({
            "date": pd.to_datetime(r["timestamp"], unit="s", utc=True)
                      .tz_convert("Asia/Shanghai").normalize().tz_localize(None),
            "open": quote.get("open"), "high": quote.get("high"),
            "low": quote.get("low"), "close": quote.get("close"),
            "volume": quote.get("volume"),
        })
        if adjclose is not None:
            df["adj_close"] = adjclose
        df.insert(0, "code", code.split(".")[0])
        # Yahoo 对停牌/无成交日返回 null，保留 NaN 而非填充
        return df.dropna(subset=["close"]).reset_index(drop=True)


def cross_check(
    em_bars: pd.DataFrame, yh_bars: pd.DataFrame, tolerance: float = 0.01
) -> pd.DataFrame:
    """交叉校验两个数据源的不复权收盘价。

    仅在 adjust=0（不复权）时有意义。复权序列因口径不同必然有差异，
    对比复权价会产生大量假警报。
    """
    a = em_bars.set_index(pd.to_datetime(em_bars["date"]))["close"].rename("eastmoney")
    b = yh_bars.set_index(pd.to_datetime(yh_bars["date"]))["close"].rename("yahoo")
    merged = pd.concat([a, b], axis=1, join="inner")
    if merged.empty:
        return merged
    merged["diff_pct"] = (merged["eastmoney"] / merged["yahoo"] - 1).abs()
    merged["mismatch"] = merged["diff_pct"] > tolerance
    return merged
