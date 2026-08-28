# FactorNexus 文档中心

> 这是所有项目文档的**入口索引**。文档按"面向读者"分层：想了解系统 → 读
> README；想理解原理 → 读 ARCHITECTURE_PRINCIPLES；想按机构标准核查 →
> 读 INSTITUTIONAL_SPEC。

---

## 文档地图

| 文档 | 面向 | 内容 | 状态 |
|---|---|---|---|
| [`../README.md`](../README.md) | 所有人 | 系统全景、快速开始、全部命令手册、项目结构、测试、版本历史 | ✅ 活跃 |
| [`ARCHITECTURE_PRINCIPLES.md`](ARCHITECTURE_PRINCIPLES.md) | 开发者/评估者 | 架构原理系统评估说明书：各层设计原理（数据/挖掘/认证/策略工厂/组合/风险预算）+ 水平对标 | ✅ 活跃（2026-08-27） |
| [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) | 决策者/开发者 | 系统改进方案（P0-P4）：验证严谨性加固 → 执行层模拟盘 → 策略深度 → 动态风险预算 → LLM 自主研究闭环；含验收标准与里程碑 | ✅ 活跃（2026-08-28） |
| [`INSTITUTIONAL_SPEC.md`](INSTITUTIONAL_SPEC.md) | 合规/审计 | 机构级规范（单一事实来源）：D1-D4 数据 / F1-F3 因子 / E1-E5 评估 / C1-C5 认证 / B1-B6 回测 / P1-P10 组合标准 | ✅ 活跃 |
| [`INSTITUTIONAL_AUDIT_2026.md`](INSTITUTIONAL_AUDIT_2026.md) | 合规/审计 | 2026 机构级审计报告（对照 SPEC 逐项核查结论） | ✅ 活跃（审计时点 2026-08-26） |
| [`STRATEGY_FACTORY_PLAN.md`](STRATEGY_FACTORY_PLAN.md) | 开发者 | 策略工厂（P24）实施方案：M1 数据层 → M2 walk-forward → M3-M4 模型池 → M5 集成 → M6 组合联动，含实施状态与实测指标 | ✅ 活跃 |
| [`UPGRADE_DESIGN_v2.0.md`](UPGRADE_DESIGN_v2.0.md) | 历史 | v2.0 升级设计文档（2026-08-26 起草）。**已全部落地实现**，仅作设计决策存档 | 📦 已归档（实现完成） |

## 建议阅读顺序

```
第一次接触项目：    README.md → ARCHITECTURE_PRINCIPLES.md（一~三章）
开始用系统挖矿：    README.md（二~五章命令手册）
做因子/模型研究：   STRATEGY_FACTORY_PLAN.md + ARCHITECTURE_PRINCIPLES.md（六章）
做合规/审计：      INSTITUTIONAL_SPEC.md → INSTITUTIONAL_AUDIT_2026.md
排查/扩展代码：     ARCHITECTURE_PRINCIPLES.md（全）→ 对应模块源码
```

## 文档维护约定

1. **单一事实来源**：SPEC 定义"标准是什么"；PRINCIPLES 解释"为什么这样设计"；
   代码 docstring 描述"怎么用"。同一指标的数值定义（如 RankIC 口径）只允许
   在一个地方定义（SPEC），其余文档引用。
2. **改动同步**：改代码 → 同步 README 命令手册 + 相关文档状态表；新增里程碑
   → README 版本历史追加一行。
3. **归档规则**：已实现的阶段性设计文档移入本索引"📦 已归档"行并加状态横幅，
   不删除（保留决策痕迹）。

## 测试与依赖

- 测试：`python -m pytest tests/ -q`（151 项，约 80 秒）
- 依赖：`requirements.txt`（由代码 import 扫描维护，2026-08-27 清理过时项）
- 环境变量：`.env.example` 模板 → 复制为 `.env` 填入 DEEPSEEK_API_KEY
