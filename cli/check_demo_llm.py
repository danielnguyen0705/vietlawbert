"""Verify the configured provider without logging credentials or error bodies."""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    names = ("PRIMARY_LLM_API_BASE", "PRIMARY_LLM_MODEL", "PRIMARY_LLM_API_KEY")
    missing = [name for name in names if not os.getenv(name, "").strip()]
    if missing:
        raise SystemExit("Missing settings: " + ", ".join(missing))
    try:
        with OpenAI(base_url=os.environ[names[0]], api_key=os.environ[names[2]], timeout=120, max_retries=0) as client:
            result = client.chat.completions.create(
                model=os.environ[names[1]], messages=[{"role": "user", "content": "Reply with the word READY."}],
                temperature=1, max_tokens=4096, stream=False,
            )
        if not (result.choices[0].message.content or "").strip():
            raise SystemExit("Provider returned no answer content")
        print("LLM_READY", os.environ[names[1]])
    except Exception as exc:
        raise SystemExit(f"LLM check failed: {type(exc).__name__}, HTTP {getattr(exc, 'status_code', 'unknown')}") from None


if __name__ == "__main__":
    main()
