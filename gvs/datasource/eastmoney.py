"""东方财富数据客户端。

接口细节与实测坑点见 docs/DATA_SOURCES.md。本模块只负责取数与规范化，
不做任何业务判断，也不静默吞掉错误 —— 静默跳过会造成隐性幸存者偏差。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import requests

from gvs import config

log = logging.getLogger(__name__)

KLINE_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
KLINE_COLUMNS = [
    "date", "open", "close", "high", "low", "volume", "amount",
    "amplitude", "pct_chg", "change", "turnover",
]

LIST_FIELDS = {
    "f12": "code", "f14": "name", "f2": "close", "f3": "pct_chg",
    "f9": "pe_ttm", "f23": "pb", "f20": "total_mv", "f21": "float_mv",
    "f115": "pe_static", "f13": "market",
}

# 主要财务指标，取自 RPT_F10_FINANCE_MAINFINADATA。
# 字段名以实测为准（接口对不存在的字段整体报错 code=9501，不会部分返回）。
# 银行/保险/证券的字段集与通用行业不同，非适用字段返回 null。
FIN_RENAME = {
    "SECURITY_CODE": "code",
    "SECURITY_NAME_ABBR": "name",
    "REPORT_DATE": "report_date",
    "NOTICE_DATE": "notice_date",
    "REPORT_DATE_NAME": "report_name",
    "REPORT_TYPE": "report_type",
    # 成长（累计口径同比）
    "TOTALOPERATEREVE": "revenue",
    "TOTALOPERATEREVETZ": "revenue_yoy",
    "PARENTNETPROFIT": "net_profit",
    "PARENTNETPROFITTZ": "net_profit_yoy",
    "KCFJCXSYJLR": "net_profit_deducted",
    "KCFJCXSYJLRTZ": "net_profit_deducted_yoy",
    # 成长（单季度口径）—— 累计数会掩盖最近一季的恶化，必须单独看
    "DJD_TOI_YOY": "q_revenue_yoy",
    "DJD_TOI_QOQ": "q_revenue_qoq",
    "DJD_DPNP_YOY": "q_net_profit_yoy",
    "DJD_DPNP_QOQ": "q_net_profit_qoq",
    "DJD_DEDUCTDPNP_YOY": "q_deducted_yoy",
    "DJD_DEDUCTDPNP_QOQ": "q_deducted_qoq",
    # 质量
    "ROEJQ": "roe",
    "ROEKCJQ": "roe_deducted",
    "ROIC": "roic",
    "ZZCJLL": "roa",
    "XSMLL": "gross_margin",
    "XSMLL_TB": "gross_margin_yoy",
    "XSJLL": "net_margin",
    "MGJYXJJE": "ocf_per_share",
    "MGJYXJJETZ": "ocf_per_share_yoy",
    "NETCASH_OPERATE_PK": "ocf_total",
    "JYXJLYYSR": "ocf_to_revenue",
    "XSJXLYYSR": "sales_cash_to_revenue",
    # 排雷。
    # 字段口径经恒等式验证：YSZKZZTS + CHZZTS == OPERATE_CYCLE
    # （002185 2026Q1：52.906 + 64.549 == 117.455），确认为应收/存货周转天数。
    # 另有字段 YSZKYYSR 未采用 —— 其值 0.0021 与"应收账款/营收"的量级不符
    # （按周转天数反推应约 0.59），口径无法证实，按宪章第四条不予使用。
    "YSZKZZTS": "receivable_days",
    "CHZZTS": "inventory_days",
    "OPERATE_CYCLE": "operating_cycle",
    "ZCFZL": "debt_ratio",
    "INTEREST_DEBT_RATIO": "interest_debt_ratio",
    "INTEREST_COVERAGE_RATIO": "interest_coverage",
    "LD": "current_ratio",
    "SD": "quick_ratio",
    # 每股与规模
    "EPSJB": "eps",
    "EPSKCJB": "eps_deducted",
    "BPS": "bps",
    "RDEXPEND": "rd_expense",
    "TOTAL_SHARE": "total_share",
}

FIN_COLUMNS = list(FIN_RENAME)


class DataSourceError(RuntimeError):
    """取数失败。调用方必须显式处理，不允许当作空数据继续。"""


def to_secid(code: str) -> str:
    """股票代码 -> 东财 secid。6/9 开头为沪市(1)，其余为深市/北交所(0)。"""
    code = code.strip().split(".")[0]
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"非法股票代码: {code!r}")
    return f"1.{code}" if code[0] in "69" else f"0.{code}"


def to_secucode(code: str) -> str:
    """股票代码 -> 带交易所后缀，用于 datacenter 接口。"""
    code = code.strip().split(".")[0]
    return f"{code}.SH" if code[0] in "69" else f"{code}.SZ"


@dataclass
class EastmoneyClient:
    session: requests.Session | None = None
    interval: float = config.REQUEST_INTERVAL
    timeout: float = config.REQUEST_TIMEOUT
    max_retries: int = config.MAX_RETRIES

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = self._new_session()
        self._last_call = 0.0

    @staticmethod
    def _new_session() -> requests.Session:
        """东财对新建 TCP 连接有节流，必须复用 keep-alive 连接。
        实测：同一 session 连续请求全部成功；每次 Connection: close 则第二次起断连。
        """
        s = requests.Session()
        s.headers.update(config.EM_HEADERS)
        return s

    def _get(self, url: str, params: dict) -> dict:
        delay = 1.0
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            gap = time.monotonic() - self._last_call
            if gap < self.interval:
                time.sleep(self.interval - gap)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                self._last_call = time.monotonic()
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError as exc:
                # 连接池中的连接已被服务端关闭，重建 session 后再试
                last_err = exc
                log.warning("连接中断 (%d/%d) %s，重建会话", attempt, self.max_retries, url)
                self.session.close()
                self.session = self._new_session()
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
            except Exception as exc:  # HTTP 错误、JSON 解析失败等
                last_err = exc
                log.warning("请求失败 (%d/%d) %s: %s", attempt, self.max_retries, url, exc)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
        raise DataSourceError(f"{url} 重试 {self.max_retries} 次仍失败") from last_err

    # ── 行情 ────────────────────────────────────────────────
    def daily_bars(
        self,
        code: str,
        start: str = "19900101",
        end: str = "20500101",
        adjust: int = 1,
    ) -> pd.DataFrame:
        """日线。adjust: 0=不复权 1=前复权 2=后复权。回测一律用前复权。"""
        payload = self._get(
            config.EM_KLINE_URL,
            {
                "secid": to_secid(code), "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": KLINE_FIELDS2, "klt": 101, "fqt": adjust,
                "beg": start.replace("-", ""), "end": end.replace("-", ""),
            },
        )
        data = payload.get("data") or {}
        klines = data.get("klines") or []
        if not klines:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        df = pd.DataFrame([k.split(",") for k in klines], columns=KLINE_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        for c in KLINE_COLUMNS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.insert(0, "code", data.get("code", code))
        df.insert(1, "name", data.get("name", ""))
        df["adjust"] = adjust
        return df.sort_values("date").reset_index(drop=True)

    # ── 股票池 ──────────────────────────────────────────────
    def universe(self, boards: list[str] | None = None, page_size: int = 100) -> pd.DataFrame:
        """全市场标的快照。注意这是**当前在市**标的，含幸存者偏差，
        用于回测的历史股票池须另行处理退市股（见 CHARTER 第三节）。"""
        fs = ",".join(boards or list(config.BOARDS))
        rows: list[dict] = []
        page, total = 1, None
        while True:
            payload = self._get(
                config.EM_LIST_URL,
                {
                    "pn": page, "pz": page_size, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                    "fid": "f3", "fs": fs, "fields": ",".join(LIST_FIELDS),
                },
            )
            data = payload.get("data") or {}
            chunk = data.get("diff") or []
            if not chunk:
                break
            rows.extend(chunk)
            total = data.get("total", 0)
            if total is not None and len(rows) >= total:
                break
            page += 1

        if not rows:
            raise DataSourceError("universe 返回空，接口可能变更")

        df = pd.DataFrame(rows).rename(columns=LIST_FIELDS)
        df = df[[c for c in LIST_FIELDS.values() if c in df.columns]]
        for c in ("close", "pct_chg", "pe_ttm", "pb", "total_mv", "float_mv", "pe_static"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["is_st"] = df["name"].str.contains("ST", case=False, na=False)
        df["board"] = df["code"].map(_board_of)
        return df.drop_duplicates("code").reset_index(drop=True)

    # ── 财务 ────────────────────────────────────────────────
    def financials(self, code: str, page_size: int = 60) -> pd.DataFrame:
        """主要财务指标历史。

        返回含 notice_date（公告日）—— 因子计算必须用它做 point-in-time 对齐，
        用 report_date 会导致提前得知财报，回测结果失真。
        """
        payload = self._get(
            config.EM_DATACENTER_URL,
            {
                "reportName": "RPT_F10_FINANCE_MAINFINADATA",
                "columns": ",".join(FIN_COLUMNS),
                "filter": f'(SECUCODE="{to_secucode(code)}")',
                "pageNumber": 1, "pageSize": page_size,
                "sortTypes": -1, "sortColumns": "REPORT_DATE",
                "source": "HSF10", "client": "PC",
            },
        )
        rows = (payload.get("result") or {}).get("data") or []
        if not rows:
            return pd.DataFrame(columns=list(FIN_RENAME.values()))

        df = pd.DataFrame(rows).rename(columns=FIN_RENAME)
        df["code"] = df["code"].astype(str).str.zfill(6)
        for c in ("report_date", "notice_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        numeric = [c for c in FIN_RENAME.values() if c not in
                   ("code", "name", "report_date", "notice_date",
                    "report_name", "report_type")]
        for c in numeric:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 公告日缺失无法做 PIT 对齐，宁可丢弃也不能猜
        missing = df["notice_date"].isna().sum()
        if missing:
            log.warning("%s: %d 期财务数据缺少公告日，已剔除", code, missing)
            df = df[df["notice_date"].notna()]
        return df.sort_values("report_date").reset_index(drop=True)


def _board_of(code: str) -> str:
    if code.startswith("688"):
        return "科创板"
    if code.startswith("300") or code.startswith("301"):
        return "创业板"
    if code.startswith(("8", "43", "92")):
        return "北交所"
    if code.startswith("6"):
        return "上证主板"
    return "深证主板"
