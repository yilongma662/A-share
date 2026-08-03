# 数据源验证记录

所有结论均来自本机实测，非文档推测。测试环境：AWS us-east-2（境外 IP `3.13.7.76`）。
测试日期：2026-08-02。

境外 IP 是本项目的重要约束 —— 部分国内数据源会拒绝境外访问，下表记录实测结果。

---

## 一、可用数据源

### 1. 东方财富 · 历史行情 `push2his.eastmoney.com`

```
GET https://push2his.eastmoney.com/api/qt/stock/kline/get
```

| 参数 | 说明 |
|---|---|
| `secid` | `0.002185`（深市）/ `1.600519`（沪市），前缀 0=深 1=沪 |
| `klt` | 101=日 102=周 103=月 |
| `fqt` | 0=不复权 1=前复权 2=后复权 |
| `beg` / `end` | `YYYYMMDD`，`end=20500101` 取到最新 |

**关键坑：必须携带 `Referer: https://quote.eastmoney.com/`，否则服务端直接断开连接
（curl 报 `Empty reply from server`），且不返回任何错误码。** 排查时极易误判为网络不通。

实测返回（002185 华天科技）：

```
2026-07-27,18.30,18.59
2026-07-28,18.00,17.47
2026-07-29,17.71,16.57
2026-07-30,16.31,14.91
2026-07-31,16.01,15.47
```

### 2. 东方财富 · 财务数据 `datacenter.eastmoney.com`

```
GET https://datacenter.eastmoney.com/securities/api/data/v1/get
    ?reportName=RPT_F10_FINANCE_MAINFINADATA
    &columns=ALL&filter=(SECUCODE="002185.SZ")
```

单次返回 **165 个字段**，覆盖利润表、资产负债表、现金流量表主要科目及衍生指标。

**含 `NOTICE_DATE`（公告日）字段 —— 这是 point-in-time 对齐的前提。**
实测 002185：`REPORT_DATE=2026-03-31`，`NOTICE_DATE=2026-04-29`，滞后 29 天。

常用字段：

| 字段 | 含义 |
|---|---|
| `REPORT_DATE` / `NOTICE_DATE` | 报告期 / 公告日 |
| `TOTALOPERATEREVE` / `TOTALOPERATEREVETZ` | 营业总收入 / 同比 % |
| `PARENTNETPROFIT` / `PARENTNETPROFITTZ` | 归母净利润 / 同比 % |
| `KCFJCXSYJLR` / `KCFJCXSYJLRTZ` | 扣非净利润 / 同比 % |
| `ROEJQ` | 加权 ROE（单季累计口径） |
| `XSMLL` / `XSJLL` | 销售毛利率 / 净利率 |
| `MGJYXJJE` | 每股经营现金流 |
| `ZCFZL` | 资产负债率 |

### 3. 东方财富 · 全市场快照 `push2delay.eastmoney.com`

```
GET https://push2delay.eastmoney.com/api/qt/clist/get
    ?fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048
```

实测返回 **5888 只** A 股标的（沪深主板 + 创业板 + 科创板 + 北交所）。

**关键坑：主域名 `push2.eastmoney.com` 对该接口返回 502，镜像 `82.push2` / `6.push2` 直接断连，
只有 `push2delay.eastmoney.com` 可用（延时行情，对研究用途无影响）。**

`fs` 板块代码：`m:0+t:6` 深主板 · `m:0+t:80` 创业板 · `m:1+t:2` 沪主板 ·
`m:1+t:23` 科创板 · `m:0+t:81+s:2048` 北交所

### 4. Yahoo Finance（交叉校验）

```
GET https://query1.finance.yahoo.com/v8/finance/chart/002185.SZ?range=6mo&interval=1d
```

实测可用，返回 `currency=CNY`，价格 15.47，与东财一致。
**用途限定为交叉校验** —— 复权口径与东财不同，不作为主数据源。

---

## 二、不可用数据源

| 数据源 | 状态 | 说明 |
|---|---|---|
| `hq.sinajs.cn` | 403 | 实时行情接口拒绝境外访问 |
| `quotes.money.163.com` | 502 | 网易财经历史数据接口不可达 |
| `stooq.com` | JS 挑战 | 需浏览器执行 PoW 验证，不适合脚本 |
| `push2.eastmoney.com` (clist) | 502 | 用 `push2delay` 替代 |
| Tushare Pro | 未测 | 需付费 token，暂不引入 |

---

## 三、工程约束

1. **所有东财请求必须带 `Referer` 与浏览器 `User-Agent`**，否则静默断连。
2. **请求需限速**。本项目默认间隔 0.15s（`config.REQUEST_INTERVAL`），
   全市场 5888 只标的全量拉取约 15 分钟。无官方限频文档，保守处理。
3. **失败必须重试并记录**，不得静默跳过 —— 静默跳过会造成隐性幸存者偏差。
4. 数据落盘为 Parquet，带溯源列，支持增量更新。

---

## 四、待验证

- [ ] 退市股历史数据是否可取（关系到幸存者偏差处理）
- [ ] 停牌日在 K 线中的表现（缺失 or 零成交量）
- [ ] 财务数据追溯调整时 `NOTICE_DATE` 的行为
- [ ] 分红送转对前复权序列的影响验证
