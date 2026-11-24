"""Shared Azure chat-completion behavior for the project's agents."""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict

from config import get_config
from src.utils.azure_cli_credential import AzureCliCredentialLite

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for agents that invoke the configured Azure chat model."""

    def __init__(self, agent_name: str, system_prompt: str = None):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        config = get_config()
        self.endpoint = config.ai_foundry_gpt4_endpoint.rstrip("/")
        self.deployment_name = config.ai_foundry_gpt4_deployment
        self.api_version = config.ai_foundry_gpt4_api_version
        self.api_key = config.ai_foundry_gpt4_key
        self.use_azure_cli = config.ai_foundry_gpt4_use_azure_cli
        self.credential = AzureCliCredentialLite() if self.use_azure_cli else None
        self.temperature = config.llm_temperature
        logger.info("[%s] Agent initialized", self.agent_name)

    def invoke_llm(self, user_message: str, system_message: str = None) -> str:
        messages = []
        effective_system = system_message or self.system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": user_message})

        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}"
            f"/chat/completions?api-version={self.api_version}"
        )
        headers = {"Content-Type": "application/json"}
        if self.use_azure_cli:
            token = self.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
            headers["Authorization"] = f"Bearer {token.token}"
        else:
            headers["api-key"] = self.api_key

        request = urllib.request.Request(
            url,
            data=json.dumps({
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 1200,
            }).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Azure chat model returned HTTP {exc.code}: {details}"
            ) from exc

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement execute()")

    def log_info(self, message: str, **kwargs):
        logger.info("[%s] %s", self.agent_name, message, extra=kwargs)

    def log_warning(self, message: str, **kwargs):
        logger.warning("[%s] %s", self.agent_name, message, extra=kwargs)

    def log_error(self, message: str, **kwargs):
        logger.error("[%s] %s", self.agent_name, message, extra=kwargs)
