"""全局配置。所有路径与可调参数集中于此，禁止在业务代码中硬编码。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("GVS_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("GVS_DATA_DIR", ROOT / "data"))
REPORT_DIR = Path(os.environ.get("GVS_REPORT_DIR", ROOT / "reports"))

RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"

# 东方财富接口。主域名 push2 对 clist 返回 502，只有 push2delay 可用，详见 docs/DATA_SOURCES.md
EM_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EM_LIST_URL = "https://push2delay.eastmoney.com/api/qt/clist/get"
EM_DATACENTER_URL = "https://datacenter.eastmoney.com/securities/api/data/v1/get"

# 不带 Referer 时东财服务端直接断开连接且不返回错误码，务必保留
EM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
}

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

REQUEST_INTERVAL = float(os.environ.get("GVS_REQUEST_INTERVAL", "0.15"))
REQUEST_TIMEOUT = float(os.environ.get("GVS_REQUEST_TIMEOUT", "20"))
MAX_RETRIES = int(os.environ.get("GVS_MAX_RETRIES", "4"))

# 东财板块代码 -> 可读名称
BOARDS = {
    "m:0+t:6": "深证主板",
    "m:0+t:80": "创业板",
    "m:1+t:2": "上证主板",
    "m:1+t:23": "科创板",
    "m:0+t:81+s:2048": "北交所",
}


@dataclass(frozen=True)
class TradingCost:
    """A 股交易成本。印花税自 2023-08-28 起为卖出单边千分之一。"""

    commission_rate: float = 0.00025      # 券商佣金，双边
    min_commission: float = 5.0           # 单笔最低佣金（元）
    stamp_duty_rate: float = 0.001        # 印花税，仅卖出
    transfer_fee_rate: float = 0.00001    # 过户费，双边
    slippage_rate: float = 0.001          # 冲击成本假设，需按策略容量调整

    def buy_cost(self, amount: float) -> float:
        return (
            max(amount * self.commission_rate, self.min_commission)
            + amount * self.transfer_fee_rate
            + amount * self.slippage_rate
        )

    def sell_cost(self, amount: float) -> float:
        return (
            max(amount * self.commission_rate, self.min_commission)
            + amount * self.stamp_duty_rate
            + amount * self.transfer_fee_rate
            + amount * self.slippage_rate
        )


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2018-01-01"
    end: str | None = None
    rebalance: str = "M"                  # M=月末 Q=季末 W=周
    top_n: int = 30
    benchmark: str = "000300"             # 沪深300
    cost: TradingCost = field(default_factory=TradingCost)
    exclude_st: bool = True
    exclude_limit_up: bool = True         # 一字涨停无法买入
    min_listed_days: int = 250            # 次新股波动异常，默认剔除上市不足一年的


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, CURATED_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
