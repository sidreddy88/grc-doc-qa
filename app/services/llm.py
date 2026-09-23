import logging
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, UpstreamServiceError

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


@dataclass(slots=True)
class LLMResult(Generic[SchemaT]):
    value: SchemaT
    usage: TokenUsage


class StructuredLLM(Protocol):
    def check_ready(self) -> None:
        """Raise ConfigurationError if the model can't be called, before any expensive work starts."""
        ...

    async def generate(self, system: str, user: str, schema: type[SchemaT]) -> LLMResult[SchemaT]: ...


class OpenAIStructuredLLM:
    """gpt-4o-mini with strict JSON-schema structured output.

    Retries and timeouts are delegated to the OpenAI client; anything that still
    fails is translated into domain errors so callers never see SDK exceptions.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chat: ChatOpenAI | None = None
        self._runnables: dict[type[BaseModel], Any] = {}

    def check_ready(self) -> None:
        if self._settings.openai_api_key is None or not self._settings.openai_api_key.get_secret_value().strip():
            raise ConfigurationError("OPENAI_API_KEY is not set; the service cannot call the language model.")

    def _runnable(self, schema: type[BaseModel]):
        self.check_ready()
        if self._chat is None:
            self._chat = ChatOpenAI(
                model=self._settings.llm_model,
                temperature=0,
                timeout=self._settings.llm_timeout_s,
                max_retries=self._settings.llm_max_retries,
                api_key=self._settings.openai_api_key,
            )
        if schema not in self._runnables:
            self._runnables[schema] = self._chat.with_structured_output(
                schema, method="json_schema", strict=True, include_raw=True
            )
        return self._runnables[schema]

    async def generate(self, system: str, user: str, schema: type[SchemaT]) -> LLMResult[SchemaT]:
        runnable = self._runnable(schema)
        try:
            output = await runnable.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            raise ConfigurationError("The OpenAI API key was rejected.") from error
        except openai.OpenAIError as error:
            logger.warning("llm_call_failed", extra={"error_type": type(error).__name__, "schema": schema.__name__})
            raise UpstreamServiceError("The language model is unavailable; please retry shortly.") from error

        parsed = output.get("parsed")
        if output.get("parsing_error") is not None or parsed is None:
            raise UpstreamServiceError("The language model returned an unusable response.")

        metadata = getattr(output.get("raw"), "usage_metadata", None) or {}
        usage = TokenUsage(
            input_tokens=int(metadata.get("input_tokens", 0)),
            output_tokens=int(metadata.get("output_tokens", 0)),
            calls=1,
        )
        return LLMResult(value=parsed, usage=usage)
