import base64
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


class Model:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv("HF_TOKEN", "")
        self.name = os.getenv("CHAT_MODEL", "moonshotai/Kimi-K3:baseten")
        self.usage = {}

    def complete(
        self, messages: list[dict], tools: list[dict] | None = None, reasoning_effort: str = "high"
    ) -> dict:
        if not self.token:
            raise RuntimeError(
                "Set HF_TOKEN in .env before preparing data or starting a conversation."
            )
        payload = {
            "model": self.name,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.2,
            "reasoning_effort": reasoning_effort,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = httpx.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Model request failed (HTTP {response.status_code}). "
                "Check your HF token, provider access, model route and account balance."
            )
        body = response.json()
        self.usage = body.get("usage", {})
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("The model returned no response. Please retry.")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError("The model reached its output limit. Ask a narrower question.")
        message = choice["message"]
        message["role"] = "assistant"
        return message

    def image(self, path: Path | str) -> dict:
        image_path = Path(path)
        contents = image_path.read_bytes()
        encoded = base64.b64encode(contents)
        encoded_text = encoded.decode("ascii")
        image_url = {"url": f"data:image/jpeg;base64,{encoded_text}"}
        return {"type": "image_url", "image_url": image_url}
