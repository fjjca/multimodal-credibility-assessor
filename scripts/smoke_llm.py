"""填入 .env 的 API key 后运行: python -m scripts.smoke_llm"""

import asyncio

from app.config import get_settings
from app.llm.client import LLMClient


async def main() -> None:
    client = LLMClient(get_settings())
    result = await client.chat(
        "decompose",
        [{"role": "user", "content": "请用一句话回答：连通测试成功。"}],
        max_tokens=50,
    )
    print(f"model={result.model} cached={result.cached} latency={result.latency_ms:.0f}ms")
    print(f"usage={result.usage}")
    print(f"content={result.content}")


if __name__ == "__main__":
    asyncio.run(main())