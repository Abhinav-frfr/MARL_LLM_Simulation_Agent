"""
agents/base_agent.py
"""

import logging
from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all LLM agents."""

    def __init__(
        self,
        name: str,
        model: str = "llama3.1:8b",
        temperature: float = 0.7
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434"
        )
        self.conversation_history = []
        logger.info(f"Initialized {name} with model: {model}")

    def call_llm(
        self, prompt: str, system_message: str = None
    ) -> str:
        try:
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            else:
                messages.append(SystemMessage(
                    content=f"You are a {self.name}. "
                            f"Be analytical and precise."
                ))
            messages.append(HumanMessage(content=prompt))
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return ""

    def process(self, input_data: Any) -> Any:
        raise NotImplementedError