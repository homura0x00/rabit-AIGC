
from provider.interface import LLMProvider
from utils.registry import Registry


class AgentEngine:
    def __init__(self, provider: LLMProvider, registry: Registry, work_dir: str, enable_thinking: bool = False) -> None:
        
        pass