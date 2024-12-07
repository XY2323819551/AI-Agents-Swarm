import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from groq import Groq, AsyncGroq
from openai import OpenAI, AsyncOpenAI
from together import Together, AsyncTogether


class ModelProvider(Enum):
    """模型提供商枚举"""
    DEEPSEEK = "DeepSeek"
    OPENAI = "OpenAI"
    GROQ = "Groq"
    TOGETHER = "Together"


@dataclass
class APIConfig:
    """API配置数据类"""
    base_url: str
    api_key: str


class ModelRegistry:
    """模型注册表"""
    
    # 模型到提供商的映射
    MODEL_PROVIDER_MAPPING = {
        # DeepSeek Models
        "deepseek-chat": ModelProvider.DEEPSEEK,

        # Groq Models
        "mixtral-8x7b-32768": ModelProvider.GROQ,
        "llama3-70b-8192": ModelProvider.GROQ,
        "llama3-groq-70b-8192-tool-use-preview": ModelProvider.GROQ,
        "llama-3.2-90b-text-preview": ModelProvider.GROQ,
        "llama-3.2-70b-versatile-preview": ModelProvider.GROQ,
        "llama-3.1-70b-versatile": ModelProvider.GROQ,
        "gemma2-9b-it": ModelProvider.GROQ,

        # Together Models
        "Qwen/Qwen2-72B-Instruct": ModelProvider.TOGETHER,
        "codellama/CodeLlama-34b-Python-hf": ModelProvider.TOGETHER,

        # OpenAI Models
        "gpt-4o": ModelProvider.OPENAI,
        "gpt-4o-mini": ModelProvider.OPENAI,
    }
    
    @classmethod
    def get_provider(cls, model_name: str) -> ModelProvider:
        """获取模型对应的提供商"""
        if model_name not in cls.MODEL_PROVIDER_MAPPING:
            raise ValueError(f"未知的模型: {model_name}")
        return cls.MODEL_PROVIDER_MAPPING[model_name]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        # 获取项目根目录
        self.root_dir = Path(__file__).resolve().parent.parent
        self.env_path = self.root_dir / '.env'
        
        # 加载环境变量
        load_dotenv(dotenv_path=self.env_path)
        
        # 初始化API配置
        self.api_configs = {
            ModelProvider.DEEPSEEK: APIConfig(
                base_url="https://api.deepseek.com",
                api_key=os.getenv("DEEPSEEK_API_KEY")
            ),
            ModelProvider.OPENAI: APIConfig(
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            ModelProvider.GROQ: APIConfig(
                base_url="https://api.groq.com",
                api_key=os.getenv("GROQ_API_KEY")
            ),
            ModelProvider.TOGETHER: APIConfig(
                base_url="https://api.together.xyz/v1",
                api_key=os.getenv("TOGETHER_API_KEY")
            ),
        }
    
    def get_api_config(self, provider: ModelProvider) -> APIConfig:
        """获取指定提供商的API配置"""
        if provider not in self.api_configs:
            raise ValueError(f"未找到提供商 {provider} 的配置")
        return self.api_configs[provider]


class LLMClientFactory:
    """LLM客户端工厂"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
    
    def _create_client(self, provider: ModelProvider, config: APIConfig, 
                      is_async: bool = False) -> Union[OpenAI, AsyncOpenAI, Groq, AsyncGroq, Together, AsyncTogether]:
        """创建客户端实例"""
        if provider == ModelProvider.DEEPSEEK or provider == ModelProvider.OPENAI:
            return AsyncOpenAI(base_url=config.base_url, api_key=config.api_key) if is_async \
                else OpenAI(base_url=config.base_url, api_key=config.api_key)
        
        elif provider == ModelProvider.GROQ:
            return AsyncGroq(base_url=config.base_url, api_key=config.api_key) if is_async \
                else Groq(base_url=config.base_url, api_key=config.api_key)
        
        elif provider == ModelProvider.TOGETHER:
            return AsyncTogether(base_url=config.base_url, api_key=config.api_key) if is_async \
                else Together(base_url=config.base_url, api_key=config.api_key)
        
        raise ValueError(f"不支持的提供商: {provider}")
    
    def get_client(self, model_name: str, is_async: bool = False) -> Any:
        """获取LLM客户端"""
        provider = ModelRegistry.get_provider(model_name)
        config = self.config_manager.get_api_config(provider)
        return self._create_client(provider, config, is_async)


class LLMResponse:
    """LLM响应处理类"""
    
    @staticmethod
    async def create_chat_completion(
        client: Any,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        is_json: bool = False,
        tools: Optional[List[Dict]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> Any:
        """创建聊天完成请求"""
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stop": stop
        }
        
        if is_json:
            params["response_format"] = {"type": "json_object"}
        
        if tools:
            params["tools"] = tools
        
        if stream:
            params["stream"] = True
        
        return await client.chat.completions.create(**params)


# 导出便捷函数
client_factory = LLMClientFactory()

async def get_model_response(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    is_json: bool = False,
    tools: Optional[List[Dict]] = None,
    stop: Optional[Union[str, List[str]]] = None
) -> str:
    """获取模型响应"""
    client = client_factory.get_client(model_name, is_async=True)
    response = await LLMResponse.create_chat_completion(
        client=client,
        model=model_name,
        messages=messages,
        temperature=temperature,
        is_json=is_json,
        tools=tools,
        stop=stop
    )
    return response.choices[0].message.content

async def get_model_response_stream(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    is_json: bool = False,
    stop: Optional[Union[str, List[str]]] = None
) -> Any:
    """获取模型流式响应"""
    client = client_factory.get_client(model_name, is_async=True)
    return await LLMResponse.create_chat_completion(
        client=client,
        model=model_name,
        messages=messages,
        temperature=temperature,
        is_json=is_json,
        stop=stop,
        stream=True
    )

def get_model_response_sync(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0,
    is_json: bool = False,
    tools: Optional[List[Dict]] = None,
    stop: Optional[Union[str, List[str]]] = None
) -> str:
    """获取模型同步响应"""
    client = client_factory.get_client(model_name, is_async=False)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"} if is_json else None,
        tools=tools,
        stop=stop
    )
    return response.choices[0].message.content

def get_model_response_with_tools(
    model_name: str = "deepseek-chat",
    messages: List[Dict[str, str]] = [],
    temperature: float = 0,
    is_json: bool = False,
    tools: Optional[List[Dict]] = None,
    stop: Optional[Union[str, List[str]]] = None
) -> Any:
    """获取带工具调用的模型响应"""
    client = client_factory.get_client(model_name, is_async=False)
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"} if is_json else None,
        tools=tools,
        stop=stop
    )
