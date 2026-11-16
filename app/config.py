from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDERS = ("zhipu", "siliconflow", "deepseek")
ROLES = ("decompose", "forensics", "prosecutor", "defender", "judge")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    model_decompose: str = "zhipu:glm-4-flash"
    model_forensics: str = "zhipu:glm-4.6v-flash"
    model_prosecutor: str = "zhipu:glm-4-flash"
    model_defender: str = "zhipu:glm-4-flash"
    model_judge: str = "zhipu:glm-4.7-flash"
    model_judge_strong: str = "zhipu:glm-4.5-flash"

    llm_cache_mode: str = "readwrite"
    llm_cache_dir: str = ".cache/llm"
    llm_max_concurrency: int = 1
    llm_max_retries: int = 6
    llm_min_interval_seconds: float = 2.0
    llm_rate_limit_backoff_seconds: float = 15.0
    llm_timeout_seconds: float = 60.0

    tavily_api_key: str = ""
    tavily_base_url: str = "https://api.tavily.com"
    retrieve_top_k: int = 3
    retrieve_max_claims: int = 5
    enable_retrieval: bool = True
    enable_debate: bool = True
    enable_escalation: bool = True
    judge_confidence_threshold: float = 0.6
    trace_dir: str = ".data/traces"

    host: str = "127.0.0.1"
    port: int = 8000

    def role_model(self, role: str) -> tuple[str, str]:
        if role not in ROLES:
            raise ValueError(f"unknown role: {role}")
        provider, _, model = getattr(self, f"model_{role}").partition(":")
        if provider not in PROVIDERS or not model:
            raise ValueError(f"invalid model config for role {role}, expected 'provider:model'")
        return provider, model

    def provider_credentials(self, provider: str) -> tuple[str, str]:
        return getattr(self, f"{provider}_base_url"), getattr(self, f"{provider}_api_key")


@lru_cache
def get_settings() -> Settings:
    return Settings()