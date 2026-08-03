"""数据管道：拉取 -> 溯源落盘 -> 增量更新。"""
from __future__ import annotations

import logging

import pandas as pd

from gvs import config
from gvs.datasource.eastmoney import EastmoneyClient
from gvs.datasource.prices import PriceService
from gvs.storage.store import Store

log = logging.getLogger(__name__)

SRC = "eastmoney"


def ingest_universe(client: EastmoneyClient | None = None, store: Store | None = None) -> pd.DataFrame:
    """全市场标的快照。

    注意：这是当前在市标的，**含幸存者偏差**。用于历史回测的股票池须补入退市股，
    否则回测天然剔除了所有归零的公司，收益被系统性高估。
    """
    client = client or EastmoneyClient()
    store = store or Store()
    df = client.universe()
    df["snapshot_date"] = pd.Timestamp.today().normalize()
    store.write(df, "universe", pd.Timestamp.today().strftime("%Y%m%d"),
                source=SRC, endpoint=config.EM_LIST_URL, dedup_on=["code"], sort_on=["code"])
    log.info("universe: %d 只标的", len(df))
    return df


def ingest_bars(
    code: str,
    client: EastmoneyClient | None = None,
    store: Store | None = None,
    adjust: int = 1,
    incremental: bool = True,
    prices: PriceService | None = None,
) -> pd.DataFrame:
    """个股日线。前复权用于回测，不复权用于展示，分数据集存放。

    经 PriceService 取数，东财限流时自动降级 Yahoo。降级数据的复权口径不同，
    因此按 _provider 分数据集存放，避免两种口径混入同一序列。
    """
    store = store or Store()
    prices = prices or PriceService(eastmoney=client or EastmoneyClient())

    start = "19900101"
    if incremental and adjust == 0:
        # 前复权序列会因分红除权整体改写历史价格，增量更新只对不复权安全
        last = store.last_date(f"bars_fq{adjust}", code)
        if last is not None:
            start = (last - pd.Timedelta(days=5)).strftime("%Y%m%d")

    df = prices.daily_bars(code, adjust=adjust, start=start)
    if df.empty:
        log.warning("%s 无行情数据（可能已退市或代码错误）", code)
        return df

    provider = str(df["_provider"].iloc[0])
    actual_adjust = int(df["adjust"].iloc[0]) if "adjust" in df else adjust
    dataset = f"bars_fq{actual_adjust}" if provider == "eastmoney" else f"bars_{provider}"
    endpoint = config.EM_KLINE_URL if provider == "eastmoney" else config.YAHOO_CHART_URL

    store.write(df, dataset, code, source=provider, endpoint=endpoint,
                dedup_on=["code", "date"], sort_on=["date"])
    return df


def ingest_financials(
    code: str, client: EastmoneyClient | None = None, store: Store | None = None
) -> pd.DataFrame:
    """个股财务。财报存在追溯调整，写入时按 (code, report_date, notice_date) 去重保留最新版本。"""
    client = client or EastmoneyClient()
    store = store or Store()
    df = client.financials(code)
    if df.empty:
        log.warning("%s 无财务数据", code)
        return df
    store.write(df, "financials", code, source=SRC, endpoint=config.EM_DATACENTER_URL,
                dedup_on=["code", "report_date", "notice_date"], sort_on=["report_date"])
    return df


def build_price_panel(
    codes: list[str],
    store: Store | None = None,
    adjust: int = 1,
    dataset: str | None = None,
) -> pd.DataFrame:
    """拼接多只股票的收盘价面板，供回测使用。缺失值保留为 NaN（代表停牌/未上市）。

    只从单一数据集读取。不同来源的复权口径不同，混入同一面板会使回测收益失真，
    因此宁可让缺失的股票缺席，也不跨源拼接。
    """
    store = store or Store()
    dataset = dataset or f"bars_fq{adjust}"
    series: dict[str, pd.Series] = {}
    missing: list[str] = []
    for c in codes:
        df = store.read(dataset, c)
        if df.empty:
            missing.append(c)
            continue
        s = df.set_index(pd.to_datetime(df["date"]))["close"]
        series[c] = s[~s.index.duplicated(keep="last")]
    if missing:
        # 显式告警：静默丢弃标的会造成隐性幸存者偏差
        log.warning("数据集 %s 中缺失 %d 只股票，未纳入面板: %s",
                    dataset, len(missing), missing[:10])
    if not series:
        raise ValueError(f"数据集 {dataset} 为空，请先执行 ingest_bars")
    return pd.DataFrame(series).sort_index()
