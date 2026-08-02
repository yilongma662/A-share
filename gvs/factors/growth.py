"""成长、质量、估值因子。

核心判断（宪章第四节）：高增速 + 低质量 = 陷阱。
营收涨 30% 而经营现金流为负，这不是成长，是风险。本模块的 quality 因子
与 growth 因子必须联合使用，单看增速的筛选结果不具备参考价值。

所有输入必须是经 gvs.factors.pit 处理过的 PIT 快照，本模块不做时点校验。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FactorResult:
    """单个因子的计算结果，含覆盖率 —— 覆盖率过低的因子不可用于排序。"""

    name: str
    values: pd.Series
    description: str
    direction: int = 1                    # 1=越大越好 -1=越小越好
    coverage: float = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.values)
        self.coverage = float(self.values.notna().sum() / n) if n else 0.0

    def rank(self, ascending: bool | None = None) -> pd.Series:
        """百分位排名，已按 direction 归一化为"越大越好"。"""
        asc = (self.direction == 1) if ascending is None else ascending
        return self.values.rank(pct=True, ascending=asc)

    def zscore(self, winsor: float = 0.01) -> pd.Series:
        """去极值后标准化。A 股财务数据极值多，不去极值会被单只股票主导。"""
        v = self.values.astype(float)
        if v.notna().sum() < 3:
            return pd.Series(np.nan, index=v.index)
        lo, hi = v.quantile(winsor), v.quantile(1 - winsor)
        v = v.clip(lo, hi)
        std = v.std(ddof=0)
        if not std or np.isnan(std):
            return pd.Series(0.0, index=v.index).where(v.notna())
        return (v - v.mean()) / std * self.direction


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df else \
        pd.Series(np.nan, index=df.index)


# ── 成长因子 ────────────────────────────────────────────────
def revenue_growth(snap: pd.DataFrame) -> FactorResult:
    return FactorResult("revenue_growth", _col(snap, "revenue_yoy"),
                        "营业总收入同比增速 %", 1)


def profit_growth(snap: pd.DataFrame) -> FactorResult:
    return FactorResult("profit_growth", _col(snap, "net_profit_yoy"),
                        "归母净利润同比增速 %", 1)


def deducted_profit_growth(snap: pd.DataFrame) -> FactorResult:
    """扣非净利增速。比归母净利更能反映主营，规避资产处置、政府补贴带来的假增长。"""
    return FactorResult("deducted_profit_growth", _col(snap, "net_profit_deducted_yoy"),
                        "扣非净利润同比增速 %", 1)


def growth_acceleration(snap: pd.DataFrame) -> FactorResult:
    """增速的加速度：营收增速 - 扣非净利增速的差。

    为负说明利润增长快于收入（可能是降本或规模效应，偏正面）；
    为正说明增收不增利（毛利被侵蚀或费用失控，偏负面）。
    """
    gap = _col(snap, "revenue_yoy") - _col(snap, "net_profit_deducted_yoy")
    return FactorResult("growth_gap", gap, "营收增速 - 扣非净利增速（越小越好）", -1)


# ── 质量因子 ────────────────────────────────────────────────
def roe(snap: pd.DataFrame) -> FactorResult:
    return FactorResult("roe", _col(snap, "roe"), "加权净资产收益率 %", 1)


def gross_margin(snap: pd.DataFrame) -> FactorResult:
    return FactorResult("gross_margin", _col(snap, "gross_margin"), "销售毛利率 %", 1)


def cash_conversion(snap: pd.DataFrame) -> FactorResult:
    """经营现金流 / 净利润。低于 1 说明利润没有现金支撑，是最重要的排雷指标之一。"""
    ocf = _col(snap, "ocf_per_share")
    eps = _col(snap, "eps")
    ratio = ocf / eps.where(eps.abs() > 1e-9)
    return FactorResult("cash_conversion", ratio, "每股经营现金流 / EPS", 1)


def leverage(snap: pd.DataFrame) -> FactorResult:
    return FactorResult("debt_ratio", _col(snap, "debt_ratio"), "资产负债率 %（越低越好）", -1)


# ── 估值因子 ────────────────────────────────────────────────
def peg(snap: pd.DataFrame, pe_col: str = "pe_ttm", growth_col: str = "net_profit_yoy") -> FactorResult:
    """PEG。增速为负时 PEG 无经济含义，置为 NaN 而非保留负值。"""
    pe = _col(snap, pe_col)
    g = _col(snap, growth_col)
    valid = (pe > 0) & (g > 0)
    val = (pe / g).where(valid)
    return FactorResult("peg", val, "PE(TTM) / 净利增速（越小越好）", -1)


def pe_ttm(snap: pd.DataFrame) -> FactorResult:
    pe = _col(snap, "pe_ttm")
    return FactorResult("pe_ttm", pe.where(pe > 0), "市盈率 TTM（越小越好，负值剔除）", -1)


# ── 复合打分 ────────────────────────────────────────────────
GROWTH_SET = (revenue_growth, deducted_profit_growth, growth_acceleration)
QUALITY_SET = (roe, gross_margin, cash_conversion, leverage)
VALUE_SET = (pe_ttm, peg)

DEFAULT_WEIGHTS = {"growth": 0.45, "quality": 0.35, "value": 0.20}


def composite_score(
    snap: pd.DataFrame,
    weights: dict[str, float] | None = None,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """三维复合打分。

    权重是**未经回测验证的初始假设**，不是最优解。
    在 gvs.backtest 给出分组单调性证据之前，不应据此做投资决策。
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    blocks = {"growth": GROWTH_SET, "quality": QUALITY_SET, "value": VALUE_SET}

    out = pd.DataFrame(index=snap.index)
    if "code" in snap:
        out["code"] = snap["code"].values
    if "report_name" in snap:
        out["report_name"] = snap["report_name"].values

    dropped: list[str] = []
    empty_blocks: list[str] = []
    block_scores: dict[str, pd.Series] = {}
    for block, funcs in blocks.items():
        parts = []
        for f in funcs:
            r = f(snap)
            out[r.name] = r.values.values
            if r.coverage < min_coverage:
                dropped.append(f"{r.name}(覆盖率{r.coverage:.0%})")
                continue
            parts.append(r.zscore())
        if parts:
            block_scores[block] = pd.concat(parts, axis=1).mean(axis=1)
        else:
            block_scores[block] = pd.Series(np.nan, index=snap.index)
            empty_blocks.append(block)
        out[f"score_{block}"] = block_scores[block].values

    # 整块因子缺失时必须重新归一化权重。
    # 若直接 fillna(0) 加权，缺失块会被当作"中性分 0"计入，
    # 结果是权重和不为 1，各维度的相对影响被悄悄改变 —— 分数看似正常，实则失真。
    live = {b: w[b] for b in blocks if b not in empty_blocks}
    wsum = sum(live.values())
    if not live:
        raise ValueError(f"所有因子块均无有效数据（剔除: {dropped}），无法打分")
    effective = {b: v / wsum for b, v in live.items()}

    total = sum(block_scores[b].fillna(0) * effective[b] for b in live)
    # 所有维度均缺失的个股不给分，避免 fillna(0) 把"无数据"伪装成"中性"
    all_nan = pd.concat([block_scores[b] for b in live], axis=1).isna().all(axis=1)
    out["score_total"] = total.where(~all_nan).values

    out.attrs["dropped_factors"] = dropped
    out.attrs["empty_blocks"] = empty_blocks
    out.attrs["weights"] = w
    out.attrs["effective_weights"] = effective
    return out.sort_values("score_total", ascending=False)
