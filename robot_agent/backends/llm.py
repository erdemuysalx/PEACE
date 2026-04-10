"""LLM backend — Ollama client with typed interface."""

from typing import Any, Optional, Protocol, runtime_checkable

from ollama import Client

from robot_agent.exceptions import InferenceError


@runtime_checkable
class LlmClient(Protocol):
    """Interface for language model backends."""

    def chat(
        self,
        messages: list[dict],
        *,
        format: Any = None,
        num_predict_override: Optional[int] = None,
    ) -> str:
        """Send messages and return the model's text response."""
        ...

    @property
    def is_ready(self) -> bool:
        """Return True if the model is available and the client is connected."""
        ...


class OllamaClient:
    """Ollama-backed language model client.

    Supports both local Ollama instances (no api_key) and Ollama cloud
    (api_key passed as Authorization: Bearer header).
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.7,
        num_predict: int = 2000,
        timeout: int = 120,
        api_key: str = "",
    ) -> None:
        """Create the Ollama client and probe whether the requested model is available."""
        self._model = model
        self._temperature = temperature
        self._num_predict = num_predict
        self._timeout = timeout
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = Client(host=endpoint, headers=headers, timeout=self._timeout)
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Return True if the configured model is listed by the Ollama server."""
        try:
            names = [m.model for m in self._client.list().models]
            return any(self._model in name for name in names)
        except Exception:
            return False

    @property
    def is_ready(self) -> bool:
        """Return True if the model was found on the Ollama server at startup."""
        return self._available

    @property
    def available_models(self) -> list[str]:
        """Return all model names currently available on the Ollama server."""
        try:
            return [m.model for m in self._client.list().models]
        except Exception:
            return []

    def chat(
        self,
        messages: list[dict],
        *,
        format: Any = None,
        num_predict_override: Optional[int] = None,
    ) -> str:
        """
        Send messages to the model and return the response text.

        Args:
            messages: List of chat messages (role/content dicts).
            format: Optional JSON schema to constrain output format.
            num_predict_override: If set, replaces the default num_predict for this call only.
                Useful for lightweight decision calls that need fewer tokens than full planning calls.

        Raises:
            InferenceError: If the Ollama API call fails.
        """
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "options": {
                "temperature": self._temperature,
                "num_predict": (
                    num_predict_override if num_predict_override is not None else self._num_predict
                ),
            },
        }
        if format is not None:
            kwargs["format"] = format

        try:
            response = self._client.chat(**kwargs)
            return response.message.content
        except Exception as e:
            raise InferenceError(f"Ollama chat failed: {e}") from e
