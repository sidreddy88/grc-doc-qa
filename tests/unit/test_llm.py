import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, UpstreamServiceError
from app.services.llm import OpenAIStructuredLLM


class Answer(BaseModel):
    text: str


class StubRunnable:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    async def ainvoke(self, messages):
        if self.error:
            raise self.error
        return self.result


def _llm_with(runnable, key="sk-test") -> OpenAIStructuredLLM:
    llm = OpenAIStructuredLLM(Settings(_env_file=None, openai_api_key=key))
    llm._runnable = lambda schema: runnable
    return llm


_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


async def test_missing_api_key_is_a_configuration_error():
    llm = OpenAIStructuredLLM(Settings(_env_file=None, openai_api_key=None))
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        await llm.generate("system", "user", Answer)


async def test_parsed_output_and_token_usage():
    raw = AIMessage(content="", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150})
    llm = _llm_with(StubRunnable({"parsed": Answer(text="ok"), "raw": raw, "parsing_error": None}))
    result = await llm.generate("system", "user", Answer)
    assert result.value.text == "ok"
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.calls) == (120, 30, 1)


async def test_timeouts_become_upstream_errors():
    llm = _llm_with(StubRunnable(error=openai.APITimeoutError(request=_REQUEST)))
    with pytest.raises(UpstreamServiceError):
        await llm.generate("system", "user", Answer)


async def test_rejected_key_becomes_configuration_error():
    response = httpx.Response(401, request=_REQUEST)
    llm = _llm_with(StubRunnable(error=openai.AuthenticationError("bad key", response=response, body=None)))
    with pytest.raises(ConfigurationError):
        await llm.generate("system", "user", Answer)


async def test_schema_mismatch_becomes_upstream_error():
    llm = _llm_with(StubRunnable({"parsed": None, "raw": AIMessage(content="{}"), "parsing_error": ValueError()}))
    with pytest.raises(UpstreamServiceError, match="unusable"):
        await llm.generate("system", "user", Answer)
