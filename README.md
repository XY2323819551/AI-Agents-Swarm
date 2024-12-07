# AI Agents Swarm

这是一个基于大语言模型的多智能体系统，通过多个专业化的 Agent 协同工作，实现复杂的交互场景。目前包含三个主要应用：

1. **智能客服系统**：通过多个专业化的客服 Agent 处理不同类型的客户需求
2. **智能诗歌创作**：结合环境信息，自动创作符合场景的诗歌
3. **AI教学对话**：模拟教师和学生的互动，实现基于论文的知识传递

## 项目结构

```
AI-Agents-Swarm/
├── agents_swarm/               # 主要的Agent实现
│   ├── agent_detail.py        # 智能客服系统
│   ├── agent_poem.py          # 智能诗歌创作系统
│   ├── agent_role_play.py     # AI教学对话系统
│   └── __init__.py
├── utils/                     # 工具函数
│   ├── agent_logger.py        # Agent日志记录
│   └── function_to_schema.py  # 函数模式转换
├── tool_pool/                 # 工具池
│   ├── arxiv.py              # arxiv论文搜索
│   ├── arxiv_pdf.py          # arxiv PDF内容提取
│   └── weather.py            # 天气信息查询
├── llm_pool/                  # LLM模型池
│   └── llm.py                # LLM接口封装
├── .env.example              # 环境变量示例
├── .gitignore               
├── Makefile                  # 项目管理脚本
└── README.md                 # 项目说明文档
```

## 功能特点

- **多Agent协作**：支持多个专业化Agent的无缝协作
- **工具调用**：集成多种实用工具，如论文搜索、天气查询等
- **日志记录**：完整的对话历史和工具调用记录
- **灵活扩展**：易于添加新的Agent和工具
- **自然交互**：支持表情符号和富文本显示

## 运行方式

1. 环境准备
```bash
# 复制环境变量示例文件
cp .env.example .env
# 编辑.env文件，填入必要的API密钥
vim .env
```

2. 运行不同的应用场景

```bash
# 运行智能客服系统
python -m agents_swarm.agent_detail

# 运行智能诗歌创作系统
python -m agents_swarm.agent_poem

# 运行AI教学对话系统
python -m agents_swarm.agent_role_play
```

## 使用示例

### 1. 智能客服系统
处理客户的售前售后问题，支持订单处理、退款等操作。

### 2. 智能诗歌创作
根据地点和天气信息自动创作诗歌，支持多种诗歌风格。

### 3. AI教学对话
模拟师生互动，基于arxiv论文进行专业知识讲解。

## 环境要求

- Python 3.8+



## 日志说明

系统会自动记录所有对话历史和工具调用记录，日志文件保存在 `agent_logs/` 目录下。


## LLM 模型池

本项目的 `llm_pool` 模块提供了一个统一的大语言模型调用接口，支持多个主流模型提供商和多种模型。

### 支持的模型提供商

- **DeepSeek**
  - deepseek-chat

- **Groq**
  - mixtral-8x7b-32768
  - llama3-70b-8192
  - llama3-groq-70b-8192-tool-use-preview
  - llama-3.2-90b-text-preview
  - llama-3.2-70b-versatile-preview
  - llama-3.1-70b-versatile
  - gemma2-9b-it

- **Together**
  - Qwen/Qwen2-72B-Instruct
  - codellama/CodeLlama-34b-Python-hf

- **OpenAI**
  - gpt-4o
  - gpt-4o-mini

### 特点

- **统一接口**: 提供统一的调用方式，屏蔽不同提供商的接口差异
- **异步支持**: 同时支持同步和异步调用
- **流式输出**: 支持流式响应，适用于实时交互场景
- **工具调用**: 支持函数调用（Function Calling）功能
- **灵活配置**: 支持通过环境变量配置不同提供商的 API 密钥

### 使用方式

```python
# 异步调用示例
from llm_pool.llm import get_model_response

async def example():
    response = await get_model_response(
        model_name="deepseek-chat",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response)

# 同步调用示例
from llm_pool.llm import get_model_response_sync

response = get_model_response_sync(
    model_name="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response)

# 流式输出示例
from llm_pool.llm import get_model_response_stream

async def stream_example():
    stream = await get_model_response_stream(
        model_name="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": "Hello"}]
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

# 工具调用示例
from llm_pool.llm import get_model_response_with_tools

response = get_model_response_with_tools(
    model_name="deepseek-chat",
    messages=[{"role": "user", "content": "Hello"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }]
)
```

## 工具池集成

本项目的 `tool_pool` 模块集成了丰富的外部工具和 API，为 Agent 提供了强大的能力扩展。

### 搜索和知识工具

- **搜索引擎**
  - DuckDuckGo 搜索
  - SerpAPI 集成
  - Stack Exchange 问答

- **百科和文献**
  - Wikipedia 查询
  - WikiData 数据库
  - PubMed 医学文献
  - Arxiv 论文搜索与下载

- **多媒体工具**
  - YouTube 视频搜索

### 开发工具

- **代码执行**
  - Python REPL
  - Bash Shell 命令
  - IPython 交互环境

- **文件操作**
  - 文件系统管理
  - 文件读写操作

### 环境和 API 工具

- **天气服务**
  - OpenWeatherMap API 集成
  - 支持全球城市天气查询

- **网络工具**
  - HTTP 请求工具
  - API 调用封装

### 工具使用示例

```python
# 使用 Arxiv 工具搜索论文
from tool_pool import ArxivTool

arxiv_tool = ArxivTool()
papers = arxiv_tool.run("Transformer architecture")
print(papers)

# 使用天气工具查询
from tool_pool import get_weather

weather = get_weather("shanghai")
print(weather)

# 使用 Python REPL
from tool_pool import PythonTool

python_tool = PythonTool()
result = python_tool.run("print('Hello, World!')")
print(result)
```

### 特点

- **统一接口**: 所有工具都继承自 `BaseTool`，提供统一的调用方式
- **错误处理**: 内置错误处理和日志记录机制
- **可扩展性**: 易于添加新的工具和 API 集成
- **类型提示**: 完整的类型注解支持
- **异步支持**: 部分工具支持异步调用


## 注意事项

1. 请确保API密钥配置正确
2. 建议使用虚拟环境运行项目
3. 部分功能可能需要网络连接

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

MIT License

