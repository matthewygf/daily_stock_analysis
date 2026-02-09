# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 Gemini API 调用逻辑
2. 利用 Google Search Grounding 获取实时新闻
3. 结合技术面和消息面生成分析报告
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from json_repair import repair_json
from datetime import date

from src.config import get_config

logger = logging.getLogger(__name__)


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    # === A股 ===
    '600519': '贵州茅台',
    '000001': '平安银行',
    '300750': '宁德时代',
    '002594': '比亚迪',
    '600036': '招商银行',
    '601318': '中国平安',
    '000858': '五粮液',
    '600276': '恒瑞医药',
    '601012': '隆基绿能',
    '002475': '立讯精密',
    '300059': '东方财富',
    '002415': '海康威视',
    '600900': '长江电力',
    '601166': '兴业银行',
    '600028': '中国石化',

    # === 美股 ===
    'AAPL': '苹果',
    'TSLA': '特斯拉',
    'MSFT': '微软',
    'GOOGL': '谷歌A',
    'GOOG': '谷歌C',
    'AMZN': '亚马逊',
    'NVDA': '英伟达',
    'META': 'Meta',
    'AMD': 'AMD',
    'INTC': '英特尔',
    'BABA': '阿里巴巴',
    'PDD': '拼多多',
    'JD': '京东',
    'BIDU': '百度',
    'NIO': '蔚来',
    'XPEV': '小鹏汽车',
    'LI': '理想汽车',
    'COIN': 'Coinbase',
    'MSTR': 'MicroStrategy',

    # === 港股 (5位数字) ===
    '00700': '腾讯控股',
    '03690': '美团',
    '01810': '小米集团',
    '09988': '阿里巴巴',
    '09618': '京东集团',
    '09888': '百度集团',
    '01024': '快手',
    '00981': '中芯国际',
    '02015': '理想汽车',
    '09868': '小鹏汽车',
    '00005': '汇丰控股',
    '01299': '友邦保险',
    '00941': '中国移动',
    '00883': '中国海洋石油',
}

# Expanded Risk Dictionary with Market Targeting
TIME_RISK_EVENTS = {
    # --- Global / US Impact (Affects almost everything) ---
    "EARNINGS": {
        "keywords": [
            "财报发布", "发布财报", "公布财报", "披露财报", # Action: Publishing
            "业绩暴雷", "业绩不及", "指引下调",             # Action: Bad news
            "即将财报", "临近财报",                         # Action: Timing
            "Earnings Release", "Report Earnings"
        ],
        "reason": "个股临近财报发布或业绩暴雷，存在隔夜跳空风险",
        "target_markets": ["ALL"]
    },

    # Optimized Fed Keywords
    "FED_DECISION": {
        "keywords": [
            "利率决议", "议息会议", "即将加息", "即将降息",
            "FOMC Meeting", "Fed Decision"
        ],
        "reason": "临近美联储利率决议，流动性预期波动极大",
        "target_markets": ["US", "HK", "CRYPTO"]
    },
    # --- China Specific (A-Shares / HK Stocks) ---
    "CN_POLICY": {
        "keywords": ["人行", "央行", "降准", "LPR", "MLF", "逆回购", "两会", "十四五"],
        "reason": "国内重大会议或监管政策发布，板块轮动风险大",
        "target_markets": ["CN", "HK"]
    },
    
    # --- US Specific (US Stocks) ---
    "US_MACRO": {
        "keywords": ["非农", "美国CPI", "美国PCE", "美股财报", "四巫日"],
        "reason": "美股核心宏观数据或交割日，由于不设涨跌幅，波动极高",
        "target_markets": ["US", "HK", "CRYPTO"]
    },
    
    # --- Crypto Specific ---
    "CRYPTO_EVENTS": {
        "keywords": ["减半", "SEC监管", "ETF审批", "链上拥堵", "Gas费"],
        "reason": "加密货币特有行业事件，存在极端波动风险",
        "target_markets": ["CRYPTO"]
    },
}

def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """
    多来源获取股票中文名称

    获取策略（按优先级）：
    1. 从传入的 context 中获取（realtime 数据）
    2. 从静态映射表 STOCK_NAME_MAP 获取
    3. 从 DataFetcherManager 获取（各数据源）
    4. 返回默认名称（股票+代码）

    Args:
        stock_code: 股票代码
        context: 分析上下文（可选）
        data_manager: DataFetcherManager 实例（可选）

    Returns:
        股票中文名称
    """
    # 1. 从上下文获取（实时行情数据）
    if context:
        # 优先从 stock_name 字段获取
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name

        # 其次从 realtime 数据获取
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    # 2. 从静态映射表获取
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    # 3. 从数据源获取
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"无法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                # 更新缓存
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"从数据源获取股票名称失败: {e}")

    # 4. 返回默认名称
    return f'股票{stock_code}'


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版

    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str

    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (>70强烈看多, >60看多, 40-60震荡, <40看空)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    decision_type: str = "hold"  # 决策类型：buy/hold/sell（用于统计）
    confidence_level: str = "中"  # 置信度：高/中/低

    # ========== 决策仪表盘 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据

    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）

    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析

    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点

    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题

    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由

    # ========== 元数据 ==========
    market_snapshot: Optional[Dict[str, Any]] = None  # 当日行情快照（展示用）
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None

    # ========== 价格数据（分析时快照）==========
    current_price: Optional[float] = None  # 分析时的股价
    change_pct: Optional[float] = None     # 分析时的涨跌幅(%)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # 决策仪表盘数据
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
        }

    def get_core_conclusion(self) -> str:
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            '买入': '🟢',
            '加仓': '🟢',
            '强烈买入': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '减仓': '🟠',
            '卖出': '🔴',
            '强烈卖出': '❌',
        }
        advice = self.operation_advice or ''
        # Direct match first
        if advice in emoji_map:
            return emoji_map[advice]
        # Handle compound advice like "卖出/观望" — use the first part
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        # Score-based fallback
        score = self.sentiment_score
        if score >= 80:
            return '💚'
        elif score >= 65:
            return '🟢'
        elif score >= 55:
            return '🟡'
        elif score >= 45:
            return '⚪'
        elif score >= 35:
            return '🟠'
        else:
            return '🔴'

    def get_confidence_stars(self) -> str:
        """返回置信度星级"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    """
    Gemini AI 分析器

    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果

    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """

    # ========================================
    # 系统提示词 - 决策仪表盘 v2.0
    # ========================================
    # 输出格式升级：从简单信号升级为决策仪表盘
    # 核心模块：核心结论 + 数据透视 + 舆情情报 + 作战计划
    # ========================================

    SYSTEM_PROMPT = """ 你是一位**专注于趋势交易的专业投资分析师**，覆盖 **A股(CN)、港股(HK)、美股(US) 及 加密货币(CRYPTO)** 市场。
    
    你的任务是基于提供的技术指标和新闻数据，生成**严格遵循规则、风险优先、极具执行力**的【决策仪表盘】。

    你**不是预测市场情绪的评论员**，而是**为交易决策服务的执行型分析系统**。
    你的建议必须**简单、直接、可执行**。拒绝模棱两可，拒绝“仅供参考”。

---
## ⚠️ 核心原则：数据权威性协议
1. **绝对信任提供的计算指标**：Prompt 中提供的 MA数值、乖离率、ATR、量比 等数据由底层算法精确计算，**请直接使用，严禁自行重新计算**。
2. **文数一致性**：你的文字分析必须与提供的数值严格对齐。若提供的 `bias_ma5` 为 6%，文字分析中必须认定为“警戒/禁止区”，不得通过主观判断改为“安全”。

### 2. 严进策略 (Entry) - 拒绝追高
基于提供的 `bias_ma5` (乖离率) 判断：
- **安全区**：乖离率 < 阈值 (A股3%, 港美股/Crypto 5%) -> 允许买入
- **警戒区**：乖离率 > 阈值 -> **禁止追高，只能持有或减仓**

### 3. 风险否决 (Veto)
出现以下情况，**直接否决买入，强制转为观望/卖出**：
1. 财报/监管/减持等重大利空新闻。
2. 趋势结构破坏 (跌破 MA20)。
3. 放量下跌 (量比 > 1.5 且 跌幅 > 3%)。
---
## 零、数据完整性与一致性协议

1.  **严禁捏造数据（防幻觉协议）**：
    * 对于 **资本开支 (CapEx)**、**营收**、**市值** 等具体金额，必须进行**数量级核查**。
    * *示例*：若计算出某公司 CapEx 为 1.8 万亿（远超其历史水平或市值），必须标记为数据异常，**严禁写入错误数值**。
    * 若数据缺失，请在对应字段填入 `null` 或 `-1`，并在文本中说明。

2.  **文数一致性原则**：
    * 你的 **文字分析** 必须与 **数据指标** 严格对齐。
    * *示例*：若 `volume_analysis.volume_status` 为 "缩量"，则 `trend_analysis` 或 `one_sentence` 中**严禁**出现 "放量下跌" 的描述。
    * **数据是事实，文字是翻译**，不得出现矛盾。
---
## 一、核心交易理念（必须严格遵守，不得擅自修改）
### 1️⃣ 趋势过滤（方向优先）

- **只在趋势成立时考虑交易**
- 多头趋势的必要条件：
  - **MA5 > MA10 > MA20**
  - **MA20 向上运行**
- 均线间距扩大优于均线粘合
- 若 **MA5 < MA10**，视为趋势走弱，禁止新开仓
- 若 **收盘价有效跌破 MA20**，视为趋势破坏，直接判定为【观望 / 卖出】

> 趋势判断基于 **日线级别**

---

### 2️⃣ 严进策略（不追高，位置决定盈亏比）

你必须严格控制买入位置，**绝不追高**。

#### 乖离率定义（基于 MA5）：
```
乖离率 = (现价 - MA5) / MA5 × 100%
```
---

### 3️⃣ 买点偏好（回撤入场）

你偏好在趋势中的**回撤结构**入场：

- **最佳买点**：缩量回踩 MA5 获得支撑
- **次优买点**：回踩 MA10 获得支撑
- **禁止买入**：
  - 放量下跌
  - 跌破 MA20
  - 高位放量滞涨

---

### 4️⃣ 量能确认（趋势是否健康）

- **缩量回调**：视为抛压减轻（加分）
- **放量上涨**：趋势确认（加分）
- **放量下跌**：
  - 若跌幅 >3% 且量比 >1.5
  - 直接触发【风险否决】

---

### 5️⃣ 结构与筹码（辅助判断）

- 获利盘 70%–85%：结构健康
- 获利盘 >90%：警惕获利回吐
- 筹码集中度 <15%：加分

---

### 6️⃣ 风险否决机制（最高优先级）

以下任意一项出现，**直接否决所有买入信号**：

- 股东 / 高管减持公告
- 监管调查 / 处罚
- 重大解禁
- 财报暴雷或业绩大幅不及预期
- 行业或政策级别利空
- 放量下跌破位

> **风险否决优先级高于所有技术评分**

---

### 7️⃣ 美股宏观与时间风险管理（必须遵守）

你必须将**美股宏观事件与时间节点**视为**高优先级风险管理因素**，其优先级等同于公司级重大利空。

#### 1. 美联储（FED）利率决议

- 在美联储利率决议公布前：
  - **公布日前 1 个交易日至公布前**：
    - 禁止新开仓
    - 已持仓者不建议加仓
- 利率决议公布当日：
  - 若结果与市场预期存在明显偏差：
    - 必须降低 `confidence_level`
    - 在 `risk_alerts` 中明确标注「利率决议不确定性风险」
- 利率决议公布后：
  - 至少等待 **1 个交易日** 再评估趋势有效性

> 你不预测利率方向，只管理事件不确定性风险。

---

#### 2. 重要宏观数据（CPI / 非农 / 失业率）

- 若重要宏观数据将在 **未来 3 个交易日内公布**：
  - 禁止新开仓
  - `time_sensitivity` 只能为「不急」
- 数据公布当日：
  - 若实际值与预测值显著偏离：
    - 标记为「宏观波动风险」
    - 降低 `confidence_level`
    - 在 `risk_alerts` 中说明「实际值 vs 预测值偏差」

> 你不解读宏观趋势，只评估短期波动风险。

---

#### 3. 个股财报发布时间（美股）

- 若个股财报将在 **未来 5 个交易日内发布**：
  - 禁止新开仓
  - 已持仓者需明确提示「财报不确定性风险」
- 财报发布当日：
  - 不给出任何买入或加仓建议
- 财报发布后：
  - 至少等待 **1 个交易日**
  - 再重新评估趋势结构与买点

> 财报属于不可控跳空风险，不纳入趋势交易入场条件。

---

## 二、评分体系（用于决策强度，不可替代否决项）

- 趋势结构：40 分
- 价格位置（乖离率）：25 分
- 量能配合：15 分
- 筹码结构：10 分
- 消息 / 情绪：10 分

### 决策分级：

- **80–100 分**：强烈买入
- **60–79 分**：买入
- **40–59 分**：观望
- **0–39 分**：卖出 / 减仓

---

## 三、输出要求（必须遵守）

你必须输出一个完整的【决策仪表盘】，并遵循以下原则：

1. **核心结论先行**：一句话直接告诉用户该做什么。必须以**动词**开头（例如："买入..."、"等待..."、"减仓..."）。
2. **区分空仓 / 持仓建议**：两者的操作逻辑完全不同，必须分开写。
3. **给出明确入场、止损、止盈逻辑**：
   - 必须基于提供的 MA 数值或整数关口给出**具体数字**。
   - 不要说 "均线附近"，要说 "**MA5 (123.45元) 附近**"。
4. **使用 ✅ ⚠️ ❌ 检查清单**：让用户一眼看清所有条件及其满足情况。
5. **风险项必须醒目标出**。
6. **盈亏比 (R/R Ratio)**：只推荐潜在收益 > 2倍潜在风险的交易。

### 额外一致性约束（必须遵守）：

- `operation_advice`、`decision_type`、`signal_type` 三者必须逻辑一致  
- 若触发风险否决机制：
  - 禁止输出任何买入或加仓建议  
  - `signal_type` 只能为 ⚠️风险警告 或 🔴卖出信号  
- 若数据不足以支持精确价格：
  - 允许使用 MA 附近区间表达  
  - 必须降低 `confidence_level` 并在 `risk_warning` 中说明原因  

---

## 四、输出格式（严格 JSON，不得增删字段）

请严格按照以下 JSON 格式输出，这是一个完整的【决策仪表盘】：

```json
{
    "stock_name": "股票中文名称",
    "sentiment_score": 0-100整数,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",

    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句话核心结论（动词开头，直接明了）",
            "signal_type": "🟢买入信号/🟡持有观望/🔴卖出信号/⚠️风险警告",
            "time_sensitivity": "立即行动/今日内/本周内/不急",
            "position_advice": {
                "no_position": "空仓者建议（是否建仓？什么价位？多少仓位？）",
                "has_position": "持仓者建议（止损在哪？止盈在哪？是否加仓？）"
            }
        },

        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均线排列状态描述",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 当前价格数值,
                "ma5": MA5数值,
                "ma10": MA10数值,
                "ma20": MA20数值,
                "bias_ma5": 乖离率百分比数值,
                "bias_status": "安全/警戒/危险",
                "support_level": 支撑位价格,
                "resistance_level": 压力位价格
            },
            "volume_analysis": {
                "volume_ratio": 量比数值,
                "volume_status": "放量/缩量/平量",
                "turnover_rate": 换手率百分比,
                "volume_meaning": "量能含义解读（如：缩量回调表示抛压减轻）"
            },
            "chip_structure": {
                "profit_ratio": 获利比例,
                "avg_cost": 平均成本,
                "concentration": 筹码集中度,
                "chip_health": "健康/一般/警惕"
            }
        },

        "intelligence": {
            "latest_news": "【最新消息】近期重要新闻摘要",
            "risk_alerts": ["风险点1：具体描述", "风险点2：具体描述"],
            "positive_catalysts": ["利好1：具体描述", "利好2：具体描述"],
            "earnings_outlook": "业绩预期分析（基于年报预告、业绩快报等）",
            "sentiment_summary": "舆情情绪一句话总结"
        },

        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "理想买入点：XX元（在MA5附近）",
                "secondary_buy": "次优买入点：XX元（在MA10附近）",
                "stop_loss": "止损位：XX元（跌破MA20或X%）",
                "take_profit": "目标位：XX元（前高/整数关口）"
            },
            "position_strategy": {
                "suggested_position": "建议仓位：X成",
                "entry_plan": "分批建仓策略描述",
                "risk_control": "风控策略描述"
            },
            "action_checklist": [
                "✅/⚠️/❌ 检查项1：多头排列",
                "✅/⚠️/❌ 检查项2：乖离率<5%",
                "✅/⚠️/❌ 检查项3：量能配合",
                "✅/⚠️/❌ 检查项4：无重大利空",
                "✅/⚠️/❌ 检查项5：筹码健康"
            ]
        }
    },

    "analysis_summary": "100字综合分析摘要",
    "key_points": "3-5个核心看点，逗号分隔",
    "risk_warning": "风险提示",
    "buy_reason": "操作理由，引用交易理念",

    "trend_analysis": "走势形态分析",
    "short_term_outlook": "短期1-3日展望",
    "medium_term_outlook": "中期1-2周展望",
    "technical_analysis": "技术面综合分析",
    "ma_analysis": "均线系统分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K线形态分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板块行业分析",
    "company_highlights": "公司亮点/风险",
    "news_summary": "新闻摘要",
    "market_sentiment": "市场情绪",
    "hot_topics": "相关热点",

    "search_performed": true/false,
    "data_sources": "数据来源说明"
}
```
---

## 五、行为约束（非常重要）

- 你是**交易决策系统**，不是荐股营销号
- 数据真实性约束：在输出基本面数据（如资本开支 CapEx、营收）前，进行常识性检查。若数值异常巨大（如 1.8万亿美金 CapEx），请再次核对单位或数据源，若无法确认则不输出。
- 致性约束：确保analysis_summary中的定性描述（如"缩量"）与volume_analysis中的定量数据（如volume_ratio < 1）完全一致。
- 不使用夸张、煽动性语言
- 不预测“必涨”“翻倍”
- 不在数据不足时强行给出具体价格
- 若数据质量不足，必须降低置信度并说明原因
- 不得在 JSON 外输出任何解释性文字  

---

## 六、你的最终目标

你的目标不是“看起来很专业”，  
而是：

> **在长期重复执行中，帮助用户避免追高、避免踩雷、只在高胜率区间出手。**

---
## 七、风险字段输出清洗协议 (Anti-False-Positive Protocol) 【至关重要】

你的 `risk_alerts` 字段会被自动化脚本读取关键词。为了防止系统误判，你必须严格遵守：

1. **仅输出“当前生效”或“即将发生”的风险**：
   - 只有当事件处于 **“禁止交易窗口期”** 内时，才允许将其写入 `risk_alerts`。
   - **禁止**在 `risk_alerts` 中描述“风险已过”或“距离风险尚早”的内容。

2. **财报与宏观事件的特殊处理**：
   - **错误写法**（会导致系统误判拦截）：
     `"risk_alerts": ["距离财报发布还有2个月，目前安全"]` 
     *(系统检测到关键词“财报”，会错误地拦截交易)*
   
   - **正确写法**（转移到利好或消息字段）：
     `"risk_alerts": []` 
     `"positive_catalysts": ["当前处于财报真空期，业绩雷风险低"]`

3. **关键词隔离原则**：
   - 如果某个风险不构成当前的阻碍，**绝对不要**在 `risk_alerts` 数组中提及该风险的专用名词（如：财报、美联储、CPI）。
---
"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 AI 分析器

        优先级：Gemini > OpenAI 兼容 API

        Args:
            api_key: Gemini API Key（可选，默认从配置读取）
        """
        config = get_config()
        self._api_key = api_key or config.gemini_api_key
        self._client = None
        self._current_model_name = None  # 当前使用的模型名称
        self._using_fallback = False  # 是否正在使用备选模型
        self._use_openai = False  # 是否使用 OpenAI 兼容 API
        self._openai_client = None  # OpenAI 客户端

        # 检查 Gemini API Key 是否有效（过滤占位符）
        gemini_key_valid = self._api_key and not self._api_key.startswith('your_') and len(self._api_key) > 10

        # 优先尝试初始化 Gemini
        if gemini_key_valid:
            try:
                self._init_model()
            except Exception as e:
                logger.warning(f"Gemini 初始化失败: {e}，尝试 OpenAI 兼容 API")
                self._init_openai_fallback()
        else:
            # Gemini Key 未配置，尝试 OpenAI
            logger.info("Gemini API Key 未配置，尝试使用 OpenAI 兼容 API")
            self._init_openai_fallback()

        # 两者都未配置
        if not self._client and not self._openai_client:
            logger.warning("未配置任何 AI API Key，AI 分析功能将不可用")

    def _init_openai_fallback(self) -> None:
        """
        初始化 OpenAI 兼容 API 作为备选

        支持所有 OpenAI 格式的 API，包括：
        - OpenAI 官方
        - DeepSeek
        - 通义千问
        - Moonshot 等
        """
        config = get_config()

        # 检查 OpenAI API Key 是否有效（过滤占位符）
        openai_key_valid = (
            config.openai_api_key and
            not config.openai_api_key.startswith('your_') and
            len(config.openai_api_key) > 10
        )

        if not openai_key_valid:
            logger.debug("OpenAI 兼容 API 未配置或配置无效")
            return

        # 分离 import 和客户端创建，以便提供更准确的错误信息
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装 openai 库，请运行: pip install openai")
            return

        try:
            # base_url 可选，不填则使用 OpenAI 官方默认地址
            client_kwargs = {"api_key": config.openai_api_key}
            if config.openai_base_url and config.openai_base_url.startswith('http'):
                client_kwargs["base_url"] = config.openai_base_url

            self._openai_client = OpenAI(**client_kwargs)
            self._current_model_name = config.openai_model
            self._use_openai = True
            logger.info(f"OpenAI 兼容 API 初始化成功 (base_url: {config.openai_base_url}, model: {config.openai_model})")
        except ImportError as e:
            # 依赖缺失（如 socksio）
            if 'socksio' in str(e).lower() or 'socks' in str(e).lower():
                logger.error(f"OpenAI 客户端需要 SOCKS 代理支持，请运行: pip install httpx[socks] 或 pip install socksio")
            else:
                logger.error(f"OpenAI 依赖缺失: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'socks' in error_msg or 'socksio' in error_msg or 'proxy' in error_msg:
                logger.error(f"OpenAI 代理配置错误: {e}，如使用 SOCKS 代理请运行: pip install httpx[socks]")
            else:
                logger.error(f"OpenAI 兼容 API 初始化失败: {e}")

    def _init_model(self) -> None:
        """
        初始化 Gemini 模型

        配置：
        - 使用 gemini-3-flash-preview 或 gemini-2.0-flash 或 gemini-1.5-flash 模型
        - 不启用 Google Search（使用外部 Tavily/SerpAPI 搜索）
        """
        try:
            from google import genai

            # 从配置获取模型名称
            config = get_config()
            model_name = config.gemini_model

            # 初始化 Client
            self._client = genai.Client(api_key=self._api_key)
            self._current_model_name = model_name
            self._using_fallback = False
            logger.info(f"Gemini API 初始化成功 (默认模型: {model_name})")

        except Exception as e:
            logger.error(f"Gemini 模型初始化失败: {e}")
            self._client = None

    def _switch_to_fallback_model(self) -> bool:
        """
        切换到备选模型

        Returns:
            是否成功切换
        """
        try:
            config = get_config()
            fallback_model = config.gemini_model_fallback

            logger.warning(f"[LLM] 切换到备选模型: {fallback_model}")
            # google-genai 不需要重新创建 client，只需在请求时更改模型名称即可
            self._current_model_name = fallback_model
            self._using_fallback = True
            logger.info(f"[LLM] 备选模型 {fallback_model} 切换成功")
            return True
        except Exception as e:
            logger.error(f"[LLM] 切换备选模型失败: {e}")
            return False

    def is_available(self) -> bool:
        """检查分析器是否可用"""
        return self._client is not None or self._openai_client is not None

    def _call_openai_api(self, prompt: str, generation_config: dict) -> str:
        """
        调用 OpenAI 兼容 API

        Args:
            prompt: 提示词
            generation_config: 生成配置

        Returns:
            响应文本
        """
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay

        def _build_base_request_kwargs() -> dict:
            kwargs = {
                "model": self._current_model_name,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": generation_config.get('temperature', config.openai_temperature),
            }
            return kwargs

        def _is_unsupported_param_error(error_message: str, param_name: str) -> bool:
            lower_msg = error_message.lower()
            return ('400' in lower_msg or "unsupported parameter" in lower_msg or "unsupported param" in lower_msg) and param_name in lower_msg

        if not hasattr(self, "_token_param_mode"):
            self._token_param_mode = {}

        max_output_tokens = generation_config.get('max_output_tokens', 8192)
        model_name = self._current_model_name
        mode = self._token_param_mode.get(model_name, "max_tokens")

        def _kwargs_with_mode(mode_value):
            kwargs = _build_base_request_kwargs()
            if mode_value is not None:
                kwargs[mode_value] = max_output_tokens
            return kwargs

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 60)
                    logger.info(f"[OpenAI] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)

                try:
                    response = self._openai_client.chat.completions.create(**_kwargs_with_mode(mode))
                except Exception as e:
                    error_str = str(e)
                    if mode == "max_tokens" and _is_unsupported_param_error(error_str, "max_tokens"):
                        mode = "max_completion_tokens"
                        self._token_param_mode[model_name] = mode
                        response = self._openai_client.chat.completions.create(**_kwargs_with_mode(mode))
                    elif mode == "max_completion_tokens" and _is_unsupported_param_error(error_str, "max_completion_tokens"):
                        mode = None
                        self._token_param_mode[model_name] = mode
                        response = self._openai_client.chat.completions.create(**_kwargs_with_mode(mode))
                    else:
                        raise

                if response and response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content
                else:
                    raise ValueError("OpenAI API 返回空响应")
                    
            except Exception as e:
                error_str = str(e)
                is_rate_limit = '429' in error_str or 'rate' in error_str.lower() or 'quota' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[OpenAI] API 限流，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                else:
                    logger.warning(f"[OpenAI] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                
                if attempt == max_retries - 1:
                    raise
        
        raise Exception("OpenAI API 调用失败，已达最大重试次数")
    
    def _call_api_with_retry(self, prompt: str, generation_config: dict) -> str:
        """
        调用 AI API，带有重试和模型切换机制
        
        优先级：Gemini > Gemini 备选模型 > OpenAI 兼容 API
        
        处理 429 限流错误：
        1. 先指数退避重试
        2. 多次失败后切换到备选模型
        3. Gemini 完全失败后尝试 OpenAI
        
        Args:
            prompt: 提示词
            generation_config: 生成配置
            
        Returns:
            响应文本
        """
        # 如果已经在使用 OpenAI 模式，直接调用 OpenAI
        if self._use_openai:
            return self._call_openai_api(prompt, generation_config)
        
        from google.genai import types
        config = get_config()
        max_retries = config.gemini_max_retries
        base_delay = config.gemini_retry_delay
        
        last_error = None
        tried_fallback = getattr(self, '_using_fallback', False)
        
        for attempt in range(max_retries):
            try:
                # 请求前增加延时（防止请求过快触发限流）
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # 指数退避: 5, 10, 20, 40...
                    delay = min(delay, 60)  # 最大60秒
                    logger.info(f"[Gemini] 第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                # 使用 google-genai SDK 调用
                response = self._client.models.generate_content(
                    model=self._current_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.SYSTEM_PROMPT,
                        temperature=generation_config.get('temperature'),
                        max_output_tokens=generation_config.get('max_output_tokens'),
                    )
                )
                
                if response and response.text:
                    return response.text
                else:
                    raise ValueError("Gemini 返回空响应")
                    
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # 检查是否是 429 限流错误
                is_rate_limit = '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower()
                
                if is_rate_limit:
                    logger.warning(f"[Gemini] API 限流 (429)，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
                    
                    # 如果已经重试了一半次数且还没切换过备选模型，尝试切换
                    if attempt >= max_retries // 2 and not tried_fallback:
                        if self._switch_to_fallback_model():
                            tried_fallback = True
                            logger.info("[Gemini] 已切换到备选模型，继续重试")
                        else:
                            logger.warning("[Gemini] 切换备选模型失败，继续使用当前模型重试")
                else:
                    # 非限流错误，记录并继续重试
                    logger.warning(f"[Gemini] API 调用失败，第 {attempt + 1}/{max_retries} 次尝试: {error_str[:100]}")
        
        # Gemini 所有重试都失败，尝试 OpenAI 兼容 API
        if self._openai_client:
            logger.warning("[Gemini] 所有重试失败，切换到 OpenAI 兼容 API")
            try:
                return self._call_openai_api(prompt, generation_config)
            except Exception as openai_error:
                logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                raise last_error or openai_error
        elif config.openai_api_key and config.openai_base_url:
            # 尝试懒加载初始化 OpenAI
            logger.warning("[Gemini] 所有重试失败，尝试初始化 OpenAI 兼容 API")
            self._init_openai_fallback()
            if self._openai_client:
                try:
                    return self._call_openai_api(prompt, generation_config)
                except Exception as openai_error:
                    logger.error(f"[OpenAI] 备选 API 也失败: {openai_error}")
                    raise last_error or openai_error
        
        # 所有方式都失败
        raise last_error or Exception("所有 AI API 调用失败，已达最大重试次数")
    
    def analyze(
        self, 
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        分析单只股票
        
        流程：
        1. 格式化输入数据（技术面 + 新闻）
        2. 调用 Gemini API（带重试和模型切换）
        3. 解析 JSON 响应
        4. 返回结构化结果
        
        Args:
            context: 从 storage.get_analysis_context() 获取的上下文数据
            news_context: 预先搜索的新闻内容（可选）
            
        Returns:
            AnalysisResult 对象
        """
        code = context.get('code', 'Unknown')
        config = get_config()
        
        # 请求前增加延时（防止连续请求触发限流）
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 请求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)
        
        # 优先从上下文获取股票名称（由 main.py 传入）
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            # 备选：从 realtime 中获取
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # 最后从映射表获取
                name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        # 如果模型不可用，返回默认结果
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary='AI 分析功能未启用（未配置 API Key）',
                risk_warning='请配置 Gemini API Key 后重试',
                success=False,
                error_message='Gemini API Key 未配置',
            )
        
        try:
            # 格式化输入（包含技术面数据和新闻）
            prompt = self._format_prompt(context, name, news_context)
            
            # 获取模型名称
            model_name = getattr(self, '_current_model_name', 'unknown')
            
            logger.info(f"========== AI 分析 {name}({code}) ==========")
            logger.info(f"[LLM配置] 模型: {model_name}")
            logger.info(f"[LLM配置] Prompt 长度: {len(prompt)} 字符")
            logger.info(f"[LLM配置] 是否包含新闻: {'是' if news_context else '否'}")
            
            # 记录完整 prompt 到日志（INFO级别记录摘要，DEBUG记录完整）
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt 预览]\n{prompt_preview}")
            logger.debug(f"=== 完整 Prompt ({len(prompt)}字符) ===\n{prompt}\n=== End Prompt ===")

            # 设置生成配置（从配置文件读取温度参数）
            config = get_config()
            generation_config = {
                "temperature": config.gemini_temperature,
                "max_output_tokens": 8192,
            }

            # 根据实际使用的 API 显示日志
            api_provider = "OpenAI" if self._use_openai else "Gemini"
            logger.info(f"[LLM调用] 开始调用 {api_provider} API...")
            
            # 使用带重试的 API 调用
            start_time = time.time()
            response_text = self._call_api_with_retry(prompt, generation_config)
            elapsed = time.time() - start_time

            # 记录响应信息
            logger.info(f"[LLM返回] {api_provider} API 响应成功, 耗时 {elapsed:.2f}s, 响应长度 {len(response_text)} 字符")
            
            # 记录响应预览（INFO级别）和完整响应（DEBUG级别）
            response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
            logger.info(f"[LLM返回 预览]\n{response_preview}")
            logger.debug(f"=== {api_provider} 完整响应 ({len(response_text)}字符) ===\n{response_text}\n=== End Response ===")
            
            # 解析响应
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            result.market_snapshot = self._build_market_snapshot(context)

            logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 评分 {result.sentiment_score}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary=f'分析过程出错: {str(e)[:100]}',
                risk_warning='分析失败，请稍后重试或手动分析',
                success=False,
                error_message=str(e),
            )
    
    def _format_prompt(
        self, 
        context: Dict[str, Any], 
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        格式化分析提示词（决策仪表盘 v2.0）
        
        包含：技术指标、实时行情（量比/换手率）、筹码分布、趋势分析、新闻
        
        Args:
            context: 技术面数据上下文（包含增强数据）
            name: 股票名称（默认值，可能被上下文覆盖）
            news_context: 预先搜索的新闻内容
        """
        code = context.get('code', 'Unknown')
        
        # 优先使用上下文中的股票名称（从 realtime_quote 获取）
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            
        today = context.get('today', {})
        
        current_date_str = date.today().isoformat()

        # ========== 构建决策仪表盘格式的输入 ==========
        prompt = f"""# 决策仪表盘分析请求

## 📊 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('date', '未知')} |
| 当前日期 | {current_date_str} |

---

## 📈 技术面数据

### 今日行情
| 指标 | 数值 |
|------|------|
| 收盘价 | {today.get('close', 'N/A')} 元 |
| 开盘价 | {today.get('open', 'N/A')} 元 |
| 最高价 | {today.get('high', 'N/A')} 元 |
| 最低价 | {today.get('low', 'N/A')} 元 |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% |
| 成交量 | {self._format_volume(today.get('volume'))} |
| 成交额 | {self._format_amount(today.get('amount'))} |

### 均线系统（关键判断指标）
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {today.get('ma5', 'N/A')} | 短期趋势线 |
| MA10 | {today.get('ma10', 'N/A')} | 中短期趋势线 |
| MA20 | {today.get('ma20', 'N/A')} | 中期趋势线 |
| 均线形态 | {context.get('ma_status', '未知')} | 多头/空头/缠绕 |
"""
        
        # 添加实时行情数据（量比、换手率等）
        if 'realtime' in context:
            rt = context['realtime']
            prompt += f"""
### 实时行情增强数据
| 指标 | 数值 | 解读 |
|------|------|------|
| 当前价格 | {rt.get('price', 'N/A')} 元 | |
| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '')} |
| **换手率** | **{rt.get('turnover_rate', 'N/A')}%** | |
| 市盈率(动态) | {rt.get('pe_ratio', 'N/A')} | |
| 市净率 | {rt.get('pb_ratio', 'N/A')} | |
| 总市值 | {self._format_amount(rt.get('total_mv'))} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |
| 60日涨跌幅 | {rt.get('change_60d', 'N/A')}% | 中期表现 |
"""
        
        # 添加筹码分布数据
        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += f"""
### 筹码分布数据（效率指标）
| 指标 | 数值 | 健康标准 |
|------|------|----------|
| **获利比例** | **{profit_ratio:.1%}** | 70-90%时警惕 |
| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 | 现价应高于5-15% |
| 90%筹码集中度 | {chip.get('concentration_90', 0):.2%} | <15%为集中 |
| 70%筹码集中度 | {chip.get('concentration_70', 0):.2%} | |
| 筹码状态 | {chip.get('chip_status', '未知')} | |
"""
        
        # 添加趋势分析结果（基于交易理念的预判）
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias_warning = "🚨 超过5%，严禁追高！" if trend.get('bias_ma5', 0) > 5 else "✅ 安全范围"
            prompt += f"""
### 趋势分析预判（基于交易理念）
| 指标 | 数值 | 判定 |
|------|------|------|
| 趋势状态 | {trend.get('trend_status', '未知')} | |
| 均线排列 | {trend.get('ma_alignment', '未知')} | MA5>MA10>MA20为多头 |
| 趋势强度 | {trend.get('trend_strength', 0)}/100 | |
| **乖离率(MA5)** | **{trend.get('bias_ma5', 0):+.2f}%** | {bias_warning} |
| 乖离率(MA10) | {trend.get('bias_ma10', 0):+.2f}% | |
| 量能状态 | {trend.get('volume_status', '未知')} | {trend.get('volume_trend', '')} |
| 系统信号 | {trend.get('buy_signal', '未知')} | |
| 系统评分 | {trend.get('signal_score', 0)}/100 | |

### 进阶技术指标 (增强)
| 指标 | 数值/状态 | 说明 |
|------|-----------|------|
| ADX (趋势强度) | {trend.get('adx', 0):.1f} | {trend.get('adx_status', '')} (>25为强趋势) |
| ATR (波动率) | {trend.get('atr', 0):.2f} | 占比 {trend.get('atr_percent', 0):.1f}% |
| K线形态 | {trend.get('candlestick_pattern', '无')} | {trend.get('candlestick_signal', '')} |

#### 系统分析理由
**买入理由**：
{chr(10).join('- ' + r for r in trend.get('signal_reasons', ['无'])) if trend.get('signal_reasons') else '- 无'}

**风险因素**：
{chr(10).join('- ' + r for r in trend.get('risk_factors', ['无'])) if trend.get('risk_factors') else '- 无'}
"""
        
        # 添加昨日对比数据
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio', 'N/A')
            prompt += f"""
### 量价变化
- 成交量较昨日变化：{volume_change}倍
- 价格较昨日变化：{context.get('price_change_ratio', 'N/A')}%
"""
        
        # 添加新闻搜索结果（重点区域）
        prompt += """
---

## 📰 舆情情报
"""
        if news_context:
            prompt += f"""
以下是 **{stock_name}({code})** 近7日的新闻搜索结果，请重点提取：
1. 🚨 **风险警报**：减持、处罚、利空
2. 🎯 **利好催化**：业绩、合同、政策
3. 📊 **业绩预期**：年报预告、业绩快报

```
{news_context}
```
"""
        else:
            prompt += """
未搜索到该股票近期的相关新闻。请主要依据技术面数据进行分析。
"""

        # 注入缺失数据警告
        if context.get('data_missing'):
            prompt += """
⚠️ **数据缺失警告**
由于接口限制，当前无法获取完整的实时行情和技术指标数据。
请 **忽略上述表格中的 N/A 数据**，重点依据 **【📰 舆情情报】** 中的新闻进行基本面和情绪面分析。
在回答技术面问题（如均线、乖离率）时，请直接说明“数据缺失，无法判断”，**严禁编造数据**。
"""

        # 明确的输出要求
        prompt += f"""
---

## ✅ 分析任务

请为 **{stock_name}({code})** 生成【决策仪表盘】，严格按照 JSON 格式输出。

### ⚠️ 重要：股票名称确认
如果上方显示的股票名称为"股票{code}"或不正确，请在分析开头**明确输出该股票的正确中文全称**。

### 重点关注（必须明确回答）：
1. ❓ 是否满足 MA5>MA10>MA20 多头排列？
2. ❓ 当前乖离率是否在安全范围内？—— 超过5%必须标注"严禁追高"
3. ❓ 量能是否配合（缩量回调/放量突破）？
4. ❓ 筹码结构是否健康？
5. ❓ 消息面有无重大利空？（减持、处罚、业绩变脸等）

### 决策仪表盘要求：
- **股票名称**：必须输出正确的中文全称（如"贵州茅台"而非"股票600519"）。
- **核心结论**：必须以动词开头（"买入"、"卖出"、"观望"）。
- **具体狙击点位**：
  - **必须给出明确数字**（精确到2位小数）。
  - 若数据不足，说明没有数据。
  - **盈亏比 (R/R)**：若潜在收益/风险 < 2.0，请在风险提示中强调。
- **检查清单**：每项用 ✅/⚠️/❌ 标记。

请输出完整的 JSON 格式决策仪表盘。"""
        
        return prompt
    
    def _format_volume(self, volume: Optional[float]) -> str:
        """格式化成交量显示"""
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 亿股"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 万股"
        else:
            return f"{volume:.0f} 股"
    
    def _format_amount(self, amount: Optional[float]) -> str:
        """格式化成交额显示"""
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 亿元"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 万元"
        else:
            return f"{amount:.0f} 元"

    def _format_percent(self, value: Optional[float]) -> str:
        """格式化百分比显示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_price(self, value: Optional[float]) -> str:
        """格式化价格显示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """构建当日行情快照（展示用）"""
        today = context.get('today', {}) or {}
        realtime = context.get('realtime', {}) or {}
        yesterday = context.get('yesterday', {}) or {}

        prev_close = yesterday.get('close')
        close = today.get('close')
        high = today.get('high')
        low = today.get('low')

        amplitude = None
        change_amount = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None
        if prev_close is not None and close is not None:
            try:
                change_amount = float(close) - float(prev_close)
            except (TypeError, ValueError):
                change_amount = None

        snapshot = {
            "date": context.get('date', '未知'),
            "close": self._format_price(close),
            "open": self._format_price(today.get('open')),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get('pct_chg')),
            "change_amount": self._format_price(change_amount),
            "amplitude": self._format_percent(amplitude),
            "volume": self._format_volume(today.get('volume')),
            "amount": self._format_amount(today.get('amount')),
        }

        if realtime:
            snapshot.update({
                "price": self._format_price(realtime.get('price')),
                "volume_ratio": realtime.get('volume_ratio', 'N/A'),
                "turnover_rate": self._format_percent(realtime.get('turnover_rate')),
                "source": getattr(realtime.get('source'), 'value', realtime.get('source', 'N/A')),
            })

        return snapshot
    
    def identify_market(self, code: str) -> str:
        """
        Identifies market type based on stock code format.
        Returns: 'CN', 'US', 'HK', 'CRYPTO', or 'UNKNOWN'
        """
        code = code.upper().strip()
        
        # 1. Crypto: Often contains USDT, BUSD, USD or is BTC/ETH
        if 'USDT' in code or 'BUSD' in code or code in ['BTC', 'ETH']:
            return 'CRYPTO'
            
        # 2. China A-Shares (CN)
        # Common formats: 60xxxx, 00xxxx, 30xxxx (6 digits)
        # Or suffix: .SS (Shanghai), .SZ (Shenzhen)
        if code.endswith(('.SS', '.SZ')):
            return 'CN'
        if re.match(r'^(60|00|30|68)\d{4}$', code):
            return 'CN'
            
        # 3. Hong Kong (HK)
        # Common formats: 00700, 09988 (5 digits), sometimes suffix .HK
        if code.endswith('.HK'):
            return 'HK'
        if re.match(r'^\d{5}$', code):
            return 'HK'
            
        # 4. US Stocks
        # Usually just letters (AAPL, TSLA) or suffix .US
        # Sometimes containing dots (BRK.B)
        if code.endswith('.US'):
            return 'US'
        if re.match(r'^[A-Z\.]+$', code): # Pure letters
            return 'US'
            
        # Default
        return 'UNKNOWN'

    def _parse_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
        ) -> AnalysisResult:
            """
            解析 Gemini 响应（决策仪表盘版） - 完整修复版
            
            改进点：
            1. 修复了逻辑覆盖 Bug：风控拦截现在拥有最高优先级。
            2. 增强了 JSON 解析的鲁棒性：防止 int() 转换失败或字典键缺失导致崩溃。
            3. 语义修正：使用 "离场回避" 代替单纯的 "清仓"，适应不同持仓状态。
            4. UI 兜底：强制修复 Dashboard 显示，确保风险提示一定会被用户看到。
            """
            try:
                # =================================================
                # 1. 文本清理与 JSON 提取
                # =================================================

                cleaned_text = response_text
                if '```json' in cleaned_text:
                    cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
                elif '```' in cleaned_text:
                    cleaned_text = cleaned_text.replace('```', '')
                
                # 尝试找到 JSON 内容
                json_start = cleaned_text.find('{')
                json_end = cleaned_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = cleaned_text[json_start:json_end]
                    # 尝试修复常见的 JSON 格式错误
                    json_str = self._fix_json_string(json_str)
                    data = json.loads(json_str)

                    # =================================================
                    # 2. 核心数据安全提取 (Safe Extraction)
                    # =================================================

                    # --- 提取情绪分 (带类型安全转换) ---
                    try:
                        sentiment_score = int(float(data.get('sentiment_score', 50)))
                    except (ValueError, TypeError):
                        sentiment_score = 50

                    # --- 提取 Dashboard 及子模块 ---
                    dashboard = data.get('dashboard', {})
                    if not isinstance(dashboard, dict): dashboard = {}
                    
                    data_perspective = dashboard.get('data_perspective', {})
                    if not isinstance(data_perspective, dict): data_perspective = {}

                    intelligence = dashboard.get('intelligence', {})
                    if not isinstance(intelligence, dict): intelligence = {}

                    # =================================================
                    # 3. 交易权限门槛计算 (Trade Permission Gates)
                    # =================================================

                    # 🛑 门槛 A: 趋势硬止损 (Trend Hard Stop)
                    trend_status = data_perspective.get('trend_status', {})
                    try:
                        trend_score = int(float(trend_status.get('trend_score', 50)))
                    except (ValueError, TypeError):
                        trend_score = 50
                    
                    # 定义：趋势破坏 = 分数低于 40
                    trend_broken = trend_score < 40

                    # 🚧 门槛 B: 环境过滤器 (Context Filters)
                    # 1. 情绪过滤器
                    sentiment_ok = sentiment_score >= 60

                    # 2. 乖离率过滤器
                    price_position = data_perspective.get('price_position', {})
                    bias_status = price_position.get('bias_status', '')
                    bias_ok = bias_status not in ['危险', '禁止']

                    # 3. 时间风险过滤器
                    risk_alerts = intelligence.get('risk_alerts', [])
                    if not isinstance(risk_alerts, list): risk_alerts = []
                    
                    risk_warning_text = data.get('risk_warning', '')

                    current_market = self.identify_market(code) # e.g., 'US' or 'CN'

                    time_risk_reasons = []
                    
                    # 拼接所有风险文本进行关键词检索
                    risk_text_pool = str(risk_warning_text) + " " + " ".join([str(x) for x in risk_alerts])

                    # 检查全局变量是否存在，防止报错
                    for event_key, event_data in TIME_RISK_EVENTS.items():
                    
                        # --- CHECK: Is this risk relevant to my market? ---
                        target_markets = event_data.get("target_markets", ["ALL"])
                        
                        is_relevant = (
                            "ALL" in target_markets or 
                            current_market == 'UNKNOWN' or 
                            current_market in target_markets
                        )
                        
                        if not is_relevant:
                            continue # Skip unrelated risks (e.g. don't check LPR for Apple)

                        # --- CHECK: Do keywords exist? ---
                        # We use 'keywords' from the dictionary to scan the AI's text
                        if any(kw in risk_text_pool for kw in event_data["keywords"]):
                            # Formatting the reason to be specific
                            prefix = f"[{current_market} Market Risk]" if current_market != 'UNKNOWN' else ""
                            time_risk_reasons.append(f"{prefix} {event_data['reason']}")
                    
                    # =================================================
                    # 4. 决策逻辑分层 (Decision Hierarchy)
                    # 优先级：趋势破坏(卖) > 环境风险(观望) > AI原始建议
                    # =================================================
                    
                    final_decision_type = ''
                    final_operation_advice = ''
                    final_confidence = ''
                    block_reasons = []
                    ALLOW_TRADE = True # 默认为真，下面逐步证伪

                    # 收集阻断原因
                    if not sentiment_ok:
                        block_reasons.append(f"市场情绪偏弱({sentiment_score})")
                    if not bias_ok:
                        block_reasons.append(f"乖离率处于{bias_status}区")
                    if time_risk_reasons:
                        block_reasons.extend(time_risk_reasons)

                    # --- 核心判定路径 ---
                    
                    if trend_broken:
                        # 🔴 路径 1: 趋势已坏 -> 强制卖出/回避
                        final_operation_advice = '卖出'
                        final_decision_type = 'sell'
                        final_confidence = '高'
                        # 插入最关键的原因到列表头部
                        block_reasons.insert(0, f"趋势结构破坏(评分{trend_score})")
                        ALLOW_TRADE = False 

                    elif block_reasons:
                        # 🟡 路径 2: 趋势尚可，但环境有风险 -> 强制观望/禁止买入
                        ai_advice = data.get('operation_advice', '持有')
                        
                        # 特殊情况：如果 AI 本身就建议卖出，我们尊重卖出建议
                        if ai_advice in ['卖出', '减仓', '强烈卖出']:
                            final_operation_advice = ai_advice
                            final_decision_type = 'sell'
                            final_confidence = data.get('confidence_level', '高')
                        else:
                            # 如果 AI 想买入或持有，由于环境风险，强制转为观望
                            final_operation_advice = '观望'
                            final_decision_type = 'hold'
                            final_confidence = '低'
                        
                        ALLOW_TRADE = False

                    else:
                        # 🟢 路径 3: 一切正常 -> 完全采纳 AI 建议
                        final_operation_advice = data.get('operation_advice', '持有')
                        final_decision_type = data.get('decision_type', '') 
                        final_confidence = data.get('confidence_level', '中')
                        ALLOW_TRADE = True

                    # --- 补全 decision_type (如果 AI 漏了) ---
                    if not final_decision_type:
                        if final_operation_advice in ['买入', '加仓', '强烈买入']:
                            final_decision_type = 'buy'
                        elif final_operation_advice in ['卖出', '减仓', '强烈卖出']:
                            final_decision_type = 'sell'
                        else:
                            final_decision_type = 'hold'

                    # =================================================
                    # 5. 数据修正与回写 (Data Correction & Write-back)
                    # =================================================

                    # 1. 修正股票名称 (优先使用 AI 识别的准确名称)
                    ai_stock_name = data.get('stock_name')
                    if ai_stock_name and (name.startswith('股票') or name == code or 'Unknown' in name):
                        name = ai_stock_name

                    # 2. 修正 Dashboard 核心结论 (确保风险被看见)
                    # 如果 Dashboard 存在，我们需要确保 core_conclusion 反映了我们的强制风控逻辑
                    if dashboard:
                        # 确保 core_conclusion 字典存在
                        if 'core_conclusion' not in dashboard or not isinstance(dashboard['core_conclusion'], dict):
                            dashboard['core_conclusion'] = {
                                'signal_type': '⚪分析中',
                                'one_sentence': '数据正在整合...',
                                'confidence_score': 50
                            }
                        
                        core = dashboard['core_conclusion']

                        # --- 场景 A: 触发“趋势破坏” (硬止损) ---
                        if trend_broken:
                            core['signal_type'] = '🔴离场回避'  # 使用中性偏空的词汇，适用所有人群
                            core['one_sentence'] = f"趋势评分过低 ({trend_score}分)，多头结构已破坏，建议立即离场或停止买入。"
                            core['time_sensitivity'] = '紧急'
                            core['block_reason'] = f"趋势硬止损触发 (评分 {trend_score} < 40)"
                            # 强制更新 dashboard 中的建议
                            core['suggestion'] = '卖出/回避'
                        
                        # --- 场景 B: 触发“环境风控” (软过滤) ---
                        elif not ALLOW_TRADE and final_decision_type != 'sell':
                            core['signal_type'] = '🟡暂缓介入'
                            # 保留 AI 原话，但加上前缀警告
                            orig_sentence = core.get('one_sentence', '')
                            core['one_sentence'] = f"【环境风险】当前胜率不高，建议观望。({orig_sentence})"
                            core['time_sensitivity'] = '不急'
                            core['block_reason'] = "; ".join(block_reasons)

                    # 3. 生成最终的 Risk Warning 文本
                    full_risk_warning = data.get('risk_warning', '')
                    if block_reasons:
                        # 将风控拦截原因加到最前面
                        full_risk_warning = f"【风控拦截】{'; '.join(block_reasons)} | " + full_risk_warning

                    # =================================================
                    # 6. 返回结果对象
                    # =================================================
                    return AnalysisResult(
                        code=code,
                        name=name,
                        # 核心指标 (使用计算后的最终值)
                        sentiment_score=sentiment_score,
                        trend_prediction=data.get('trend_prediction', '震荡'),
                        operation_advice=final_operation_advice,
                        decision_type=final_decision_type,
                        confidence_level=final_confidence,
                        # 决策仪表盘
                        dashboard=dashboard,
                        # 原始分析数据透传
                        trend_analysis=data.get('trend_analysis', ''),
                        short_term_outlook=data.get('short_term_outlook', ''),
                        medium_term_outlook=data.get('medium_term_outlook', ''),
                        technical_analysis=data.get('technical_analysis', ''),
                        ma_analysis=data.get('ma_analysis', ''),
                        volume_analysis=data.get('volume_analysis', ''),
                        pattern_analysis=data.get('pattern_analysis', ''),
                        fundamental_analysis=data.get('fundamental_analysis', ''),
                        sector_position=data.get('sector_position', ''),
                        company_highlights=data.get('company_highlights', ''),
                        news_summary=data.get('news_summary', ''),
                        market_sentiment=data.get('market_sentiment', ''),
                        hot_topics=data.get('hot_topics', ''),
                        # 综合
                        analysis_summary=data.get('analysis_summary', '分析完成'),
                        key_points=data.get('key_points', ''),
                        risk_warning=full_risk_warning,
                        buy_reason=data.get('buy_reason', ''),
                        # 元数据
                        search_performed=data.get('search_performed', False),
                        data_sources=data.get('data_sources', '技术面数据'),
                        success=True,
                    )
                else:
                    # 没找到 JSON
                    logger.warning(f"无法从响应中提取 JSON，使用原始文本分析")
                    return self._parse_text_response(response_text, code, name)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}，尝试从文本提取")
                return self._parse_text_response(response_text, code, name)
            except Exception as e:
                logger.error(f"解析过程发生未知错误: {e}")
                # 发生未知错误时兜底
                return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        import re
        
        # 移除注释
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # 修复尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保布尔值是小写
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        # fix by json-repair
        json_str = repair_json(json_str)
        
        return json_str
    
    def _parse_text_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """从纯文本响应中尽可能提取分析信息"""
        # 尝试识别关键词来判断情绪
        sentiment_score = 50
        trend = '震荡'
        advice = '持有'
        
        text_lower = response_text.lower()
        
        # 简单的情绪识别
        positive_keywords = ['看多', '买入', '上涨', '突破', '强势', '利好', '加仓', 'bullish', 'buy']
        negative_keywords = ['看空', '卖出', '下跌', '跌破', '弱势', '利空', '减仓', 'bearish', 'sell']
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = '看多'
            advice = '买入'
            decision_type = 'buy'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = '看空'
            advice = '卖出'
            decision_type = 'sell'
        else:
            decision_type = 'hold'
        
        # 截取前500字符作为摘要
        summary = response_text[:500] if response_text else '无分析结果'
        
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            decision_type=decision_type,
            confidence_level='低',
            analysis_summary=summary,
            key_points='JSON解析失败，仅供参考',
            risk_warning='分析结果可能不准确，建议结合其他信息判断',
            raw_response=response_text,
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        批量分析多只股票
        
        注意：为避免 API 速率限制，每次分析之间会有延迟
        
        Args:
            contexts: 上下文数据列表
            delay_between: 每次分析之间的延迟（秒）
            
        Returns:
            AnalysisResult 列表
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# 便捷函数
def get_analyzer() -> GeminiAnalyzer:
    """获取 Gemini 分析器实例"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟上下文数据
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': '多头排列 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI 分析测试 ===")
        result = analyzer.analyze(test_context)
        print(f"分析结果: {result.to_dict()}")
    else:
        print("Gemini API 未配置，跳过测试")
