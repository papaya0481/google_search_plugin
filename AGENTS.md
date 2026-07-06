# feat/search-subagent 分支实施计划

本文件仅用于 `feat/search-subagent` 功能分支，记录 Tavily Search Subagent 的实施约束。

## 范围

- 只修改本插件，不修改 MaiBot 主程序。
- 外层 Planner 仍只通过公开的 `web_search` 进入搜索流程。
- 第一版仅在实际命中 Tavily 且显式开启实验开关时启用私有 subagent；其他引擎、URL 直访、图片搜索和缩写翻译保持现状。

## 行为

- 新增默认关闭的 `tavily_subagent_enabled`。
- 关闭时保留原有 `tavily_include_answer`、`tavily_include_raw_content` 行为。
- 开启时忽略上述两个配置值，Tavily Search 强制使用 `include_answer=false`、`include_raw_content=false`，并保留 `title/url/content/score`。
- subagent 只获得两个不注册为公开 `@Tool` 的私有动作：
  - `extract(result_ids)`：只能选择当前搜索结果中的 1～3 个唯一 ID。
  - `finish(mode="summary"|"results", ...)`：总结必须选择来源；资料模式由插件根据结果 ID 生成。
- Extract 默认使用 `basic`、原问题作为 `query`、每来源 3 个相关 chunk、最多调用 2 次。
- subagent 默认最多 4 个成功决策轮。LLM 失败、超时、空响应和响应解析失败使用独立重试预算，失败后重试 2 次，不消耗决策轮。
- Extract 超时、限流、服务端错误和网络异常独立重试 2 次，不额外消耗决策轮；耗尽后将明确错误交回 subagent。
- 决策轮耗尽后必须进入不计轮次的强制总结阶段，只允许 `finish(summary)`，允许总结为“无有关信息”。强制总结重试也耗尽后，才 fallback 初始 Tavily 资料列表。

## 配置与兼容

- `config_version` 从 `4.0.0` 提升到 `4.0.1`，插件版本保持 `4.0.1`。
- manifest 增加 `llm.generate_with_tools` 能力并提高 SDK 最低兼容版本。
- 不新增 ConfigUpgradeHook。

## 提交与验证

- 按计划为 AGENTS、Tavily Extract、配置、subagent、测试和文档分别创建可审查的原子提交，不压缩提交。
- 每次提交前运行对应定向测试和 `git diff --check`。
- 最终使用 `uv` 运行插件定向 pytest、语法检查和 `git diff --check`；不运行需要真实 Tavily API key 的联网测试。
