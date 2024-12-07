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

7. poc_swarm_stage8: 使用中文进行测试
8. poc_swarm_stage9: 基于8进行multi_agent的demo
