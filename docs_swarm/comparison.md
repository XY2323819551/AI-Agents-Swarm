# POC Swarm 演进分析报告

本报告分析了 POC Swarm 从 Stage 1 到 Stage 9 的演进过程，每个阶段的新增特性和改进。

## Stage 1: 基础对话系统
- 实现了最基本的对话循环系统
- 使用预定义的 system_message 作为客服例程
- 简单的消息处理和响应机制
- 特点：简单性和鲁棒性，类似状态机的条件判断

## Stage 2: 函数调用能力
相比 Stage 1 新增：
- 引入了函数调用（Function Calling）能力
- 添加了具体工具函数：`look_up_item` 和 `execute_refund`
- 实现了工具调用的执行机制 `execute_tool_call`
- 改进了消息处理循环，支持工具调用

## Stage 3: Agent 概念引入
相比 Stage 2 新增：
- 引入了 Agent 类的概念
- 从静态的 system_message 转向动态的 Agent instructions
- 支持不同 Agent 的工具集
- 实现了基础的 Agent 切换机制（手动切换）

## Stage 4: Agent 自动切换
相比 Stage 3 新增：
- 实现了 Agent 之间的自动切换机制
- 引入了 Response 类来管理返回结果
- 工具函数可以返回 Agent 对象来触发切换
- 完善了 Agent 切换的消息处理

## Stage 5: 上下文管理
相比 Stage 4 新增：
- 增加了上下文变量管理
- 改进了 Agent 的配置和初始化
- 增强了消息处理的灵活性
- 支持更复杂的对话场景

## Stage 6: 高级特性
相比 Stage 5 新增：
- 添加了并行工具调用支持
- 引入了工具选择机制（tool_choice）
- 增加了最大对话轮数限制
- 改进了错误处理和日志记录

## Stage 7: 中文支持（GPT-4）
相比之前版本新增：
- 添加了中文支持
- 使用 GPT-4 作为底层模型
- 优化了中文场景下的对话体验
- 改进了中文提示词设计

## Stage 8: 深度学习模型集成（Deepseek）
相比 Stage 8 新增：
- 集成了 Deepseek 模型
- 支持多模型切换
- 优化了模型响应处理
- 增强了系统的扩展性


## Conclusion
1. poc_swarm_stage1: 依靠system message 和 routine 来完成任务，没有函数调用
2. poc_swarm_stage2: 依靠system message 和 routine 来完成任务，有函数调用
3. poc_swarm_stage3: system_message从预先定义的routine到动态的Agent的instructions（手动handoff）
4. poc_swarm_stage4: system_message从预先定义的routine到动态的Agent的instructions（agent自动handoff）
5. poc_swarm_stage5: system_message从预先定义的routine到动态的Agent的instructions（agent自动handoff，more agents）
6. poc_swarm_stage6: system_message从预先定义的routine到动态的Agent的instructions（agent自动handoff，more agents，使用不同的model进行测试，并进行对比，支持的模型有：
    - "gpt-4o", "gpt-4o-mini"（openai sdk）
    - "deepseek-chat"（openai sdk）
    - "mixtral-8x7b-32768"（groq sdk）
    - "Qwen/Qwen2-72B-Instruct"（together sdk）


这个演进过程展示了如何从一个简单的对话系统逐步构建成一个功能完善的多 Agent 系统，每个阶段都有明确的改进重点和新增特性。 
