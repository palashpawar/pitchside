"""Model providers for the agents, selected with PITCHSIDE_LLM.

offline        built-in grounded templates (default; no network, no keys)
azure          Azure OpenAI / Azure AI Foundry deployment
               AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_KEY (or Entra ID via az login)
foundry_local  Foundry Local on-device runtime (OpenAI-compatible)
               FOUNDRY_LOCAL_ENDPOINT (e.g. http://localhost:5273/v1), FOUNDRY_LOCAL_MODEL
github         GitHub Models (Azure-hosted), GITHUB_TOKEN, GITHUB_MODEL (default openai/gpt-4.1-mini)
openai         any OpenAI-compatible endpoint: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

CACHE_PATH = Path(os.environ.get("PITCHSIDE_CACHE", Path(__file__).resolve().parents[2] / "data" / "llm_cache.jsonl"))


def provider() -> str:
    return os.environ.get("PITCHSIDE_LLM", "offline").lower()


def chat_client():
    """Return an Agent Framework chat client, or None for offline mode."""
    p = provider()
    if p == "offline":
        return None
    from agent_framework.openai import OpenAIChatCompletionClient

    if p == "azure":
        kwargs = {"model": os.environ["AZURE_OPENAI_DEPLOYMENT"], "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
                  "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")}
        if os.environ.get("AZURE_OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
        else:
            from azure.identity import DefaultAzureCredential
            kwargs["credential"] = DefaultAzureCredential()
        return OpenAIChatCompletionClient(**kwargs)
    if p == "foundry_local":
        return OpenAIChatCompletionClient(model=os.environ.get("FOUNDRY_LOCAL_MODEL", "phi-4-mini"),
                                          base_url=os.environ.get("FOUNDRY_LOCAL_ENDPOINT", "http://localhost:5273/v1"),
                                          api_key="not-needed")
    if p == "github":
        return OpenAIChatCompletionClient(model=os.environ.get("GITHUB_MODEL", "openai/gpt-4.1-mini"),
                                          base_url="https://models.github.ai/inference",
                                          api_key=os.environ["GITHUB_TOKEN"])
    if p == "openai":
        return OpenAIChatCompletionClient(model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                                          base_url=os.environ.get("OPENAI_BASE_URL"),
                                          api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(f"unknown PITCHSIDE_LLM provider: {p}")


class ResponseCache:
    """Disk cache of model responses so a seeded match replays identically and cheaply."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.mem: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    self.mem[row["k"]] = row["v"]
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]

    def get(self, k: str) -> str | None:
        return self.mem.get(k)

    def put(self, k: str, v: str) -> None:
        self.mem[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))
