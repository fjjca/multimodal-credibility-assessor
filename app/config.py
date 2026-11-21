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
    llm_timeout_seconds: float = 60.0
    llm_max_concurrency: int = 1
    llm_max_retries: int = 3
    # 按角色覆盖重试次数，如 {"forensics": 2}；免费档配额限流下重试太多只是拖时间
    llm_retry_budgets: dict[str, int] = {}
    llm_min_interval_seconds: float = 2.0
    llm_rate_limit_backoff_seconds: float = 15.0
    llm_rate_limit_backoff_max_seconds: float = 60.0
    # 熔断：同一模型连续 429 达阈值即冷却，冷却期内直接跳过该模型（进程内状态）
    llm_circuit_threshold: int = 3
    llm_circuit_cooldown_seconds: float = 300.0
    # 每角色的备胎模型链，如 {"forensics": ["siliconflow:PaddlePaddle/PaddleOCR-VL-1.5"]}
    llm_role_fallbacks: dict[str, list[str]] = {}

    tavily_api_key: str = ""
    tavily_base_url: str = "https://api.tavily.com"
    retrieve_top_k: int = 3
    retrieve_max_claims: int = 5
    enable_retrieval: bool = True
    enable_debate: bool = True
    enable_escalation: bool = True
    enable_langgraph: bool = False
    enable_checkpoint: bool = False
    checkpoint_path: str = ".data/langgraph_checkpoints.sqlite"
    enable_hitl: bool = False
    enable_calibration: bool = False
    calibrator_path: str = ".data/calibrator.json"
    debate_max_rounds: int = 1
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

    def retry_budget(self, role: str) -> int:
        return max(0, int(self.llm_retry_budgets.get(role, self.llm_max_retries)))

    def role_chain(self, role: str) -> list[str]:
        """主模型 + 备胎链：限流或熔断时按顺序降级到下一个候选。"""
        provider, model = self.role_model(role)
        chain = [f"{provider}:{model}"]
        for fb in self.llm_role_fallbacks.get(role, []):
            cand = (fb or "").strip()
            if not cand or cand in chain:
                continue
            p, _, m = cand.partition(":")
            if p not in PROVIDERS or not m:
                continue
            chain.append(cand)
        return chain

    @staticmethod
    def split_candidate(candidate: str) -> tuple[str, str]:
        provider, _, model = candidate.partition(":")
        return provider, model


@lru_cache
def get_settings() -> Settings:
    return Settings()