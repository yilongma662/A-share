# GVS Infinity

**Growth Valuation System** — 机构级 A 股成长股研究系统

一套可复现、可证伪、可回测的 A 股成长股研究基础设施。

> 项目的最高约束是 [`docs/CHARTER.md`](docs/CHARTER.md)。
> 任何代码与结论若与宪章冲突，以宪章为准。

---

## 设计原则

| 原则 | 工程落实 |
|---|---|
| 结论须溯源 | `Store.write()` 强制注入 `_source` / `_endpoint` / `_fetched_at`，无开关可绕过 |
| 模型须可回测 | 不能被 `gvs.backtest` 验证的策略不进生产 |
| 严禁前视偏差 | 财务数据一律按 `NOTICE_DATE`（公告日）对齐，`gvs.factors.pit` 提供断言工具 |
| 数据不足即声明 | `FactorResult.coverage` 记录覆盖率，诊断报告设"证据不足"专区 |
| 不迎合结论 | 风险标记与阈值假设一并输出，未经回测的打分明确标注不可用于决策 |

---

## 快速开始

```bash
pip install -e ".[dev]"

python -m gvs.cli health                 # 检查数据源可用性
python -m gvs.cli diagnose 002185        # 个股诊断
python -m gvs.cli universe               # 更新全市场标的表（5888 只）
python -m gvs.cli fetch 002185 600519    # 拉取行情与财务并落盘
python -m gvs.cli screen --top 20        # 成长股筛选
```

诊断输出示例（真实数据，2026-07-31）：

```
华天科技 (002185)   截至 2026-07-31   收盘 15.47   [行情源: eastmoney]
技术面
  [-] 均线位置      MA5:16.60 MA10:17.29 MA20:19.89 ... — 低于 5/6 条均线
  [-] MACD         DIFF -1.10  DEA -0.41  柱 -1.37
基本面
  [·] 最新报告期     2026一季报  公告于 2026-04-29  — 数据滞后 93 天
  [+] 营收同比(累计)  +34.5%   (48.00 亿)
  [+] 增速趋势      +17% → +21% → +23% → +34%
  [-] ROE(加权)    0.48%
风险标记
  ! 高增长低回报：营收 +34.5% 但 ROE 仅 0.48%，增长未转化为股东回报
证据不足
  ? 陷阱识别阈值为经验假设，尚未经回测验证，不应作为独立决策依据
```

---

## 模块结构

```
gvs/
  config.py              全局配置、交易成本模型、回测参数
  datasource/
    eastmoney.py         东财客户端（行情 / 财务 / 全市场）
    yahoo.py             Yahoo 备用源与交叉校验
    prices.py            多源故障转移 + 熔断器
  storage/store.py       Parquet 存储，强制溯源，支持增量
  factors/
    pit.py               point-in-time 对齐（防前视偏差）
    growth.py            成长 / 质量 / 估值因子与复合打分
  backtest/
    engine.py            回测引擎（T+1、停牌、交易成本）
    metrics.py           绩效指标与分组单调性检验
  research/diagnose.py   个股诊断
  pipeline.py            数据管道
  cli.py                 命令行入口
```

---

## 数据源

完整实测记录见 [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)。要点：

- **行情**：`push2his.eastmoney.com`，必须携带 `Referer` 请求头，否则服务端静默断连
- **财务**：`datacenter.eastmoney.com`，165 个字段，**含公告日**
- **全市场**：`push2delay.eastmoney.com`，5888 只标的（主域名 `push2` 对该接口返回 502）
- **备用**：Yahoo Finance，用于东财限流时降级与交叉校验

**东财对单 IP 有主机级累积限流**，触发后连续数分钟直接断开连接且不返回错误码。
因此 `PriceService` 实现了多源故障转移与熔断器 —— 熔断使批量任务从约 17 秒/只降至约 1 秒/只。

降级数据的复权口径与东财不同，按 `_provider` 分数据集存放，`build_price_panel`
只从单一数据集读取，宁可让标的缺席也不跨源拼接。

---

## 已处理与未处理的偏差

| 陷阱 | 状态 |
|---|---|
| 前视偏差（财务） | 已处理：`NOTICE_DATE` 对齐 + `assert_no_lookahead` 断言 |
| 前视偏差（成交） | 已处理：T+1 成交 |
| 停牌 | 已处理：价格缺失视为不可交易，持仓延续 |
| 交易成本 | 已处理：佣金 + 印花税 + 过户费 + 冲击成本 |
| 复权口径混用 | 已处理：按 `_provider` 隔离数据集 |
| **幸存者偏差** | **未处理**：股票池取自当前在市标的，缺退市股 |
| **涨跌停** | **未处理**：一字板无法成交，回测会高估收益 |
| 次新股 | 部分处理：`min_listed_days` 可配置 |

未处理项会**系统性高估**回测收益。在补齐之前，任何回测结果都不应作为实盘依据。

---

## 测试

```bash
python -m pytest tests/ -q
```

重点覆盖 point-in-time 对齐的边界条件 —— 前视偏差不会让程序崩溃，
只会让回测结果变好看，因此必须靠测试守住。

---

## 状态

阶段一（数据基础设施）已完成。在幸存者偏差与涨跌停处理补齐、
且因子通过分组单调性检验之前，本系统不输出任何选股结论。

本仓库所有输出均为数据陈述与研究工具，不构成投资建议。
