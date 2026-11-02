"""Grounded answer generation through LangChain chat models.

Providers: Claude via `langchain-anthropic`, Azure AI Foundry / Azure OpenAI deployments via
`langchain-openai`, and an offline extractive fake used in CI. Keys are bring-your-own and read
from the environment.
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import LLMConfig
from .store import Hit
from .text import tokenize

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages. Cite every claim with the "
    "passage number in square brackets, like [2]. If the passages do not contain the answer, "
    "say you don't know. Do not use outside knowledge."
)


def build_messages(question: str, hits: list[Hit]) -> list[BaseMessage]:
    ctx = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, start=1))
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"Context:\n{ctx}\n\nQuestion: {question}"),
    ]


class ExtractiveFakeChatModel(BaseChatModel):
    """Offline stand-in: answers with the content of the passage that best overlaps the
    question, cited. Deterministic, so API tests can assert on it."""

    @property
    def _llm_type(self) -> str:
        return "extractive-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw: Any):  # noqa: ANN001
        prompt = str(messages[-1].content)
        question = prompt.rsplit("Question:", 1)[-1]
        q = set(tokenize(question))
        best, best_score = "I don't know.", 0
        for m in re.finditer(r"^\[(\d+)\] (.*?)(?=^\[\d+\] |\nQuestion:|\Z)", prompt, re.M | re.S):
            n, body = m.group(1), m.group(2)
            # first line of a passage is its "Title > Section" breadcrumb, not content
            content = " ".join(line.strip() for line in body.splitlines()[1:] if line.strip())
            score = len(q & set(tokenize(content)))
            if score > best_score:
                best, best_score = f"{content} [{n}]", score
        return ChatResult(generations=[ChatGeneration(message=AIMessage(best))])


def make_chat_model(cfg: LLMConfig) -> BaseChatModel:
    if cfg.provider == "fake":
        return ExtractiveFakeChatModel()
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("provider=anthropic needs ANTHROPIC_API_KEY in the environment")
        return ChatAnthropic(
            model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens, **cfg.params
        )
    if cfg.provider == "azure_foundry":
        from langchain_openai import AzureChatOpenAI

        missing = [
            v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY") if not os.environ.get(v)
        ]
        if missing:
            raise RuntimeError(f"provider=azure_foundry needs {', '.join(missing)}")
        return AzureChatOpenAI(
            azure_deployment=cfg.model,
            api_version=cfg.params.get("api_version", "2024-10-21"),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    raise ValueError(f"unknown provider {cfg.provider!r}")


_CITE = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str, n_hits: int) -> list[int]:
    """1-based passage numbers the model actually cited, dropping out-of-range ones."""
    seen: list[int] = []
    for m in _CITE.finditer(answer):
        i = int(m.group(1))
        if 1 <= i <= n_hits and i not in seen:
            seen.append(i)
    return seen
