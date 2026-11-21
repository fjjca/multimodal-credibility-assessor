from app.config import Settings
from scripts.probe_models import candidates


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_candidates_cover_all_roles_and_dedupe_models():
    rows = candidates(_settings())
    models = [m for _, _, m in rows]
    assert "zhipu:glm-4-flash" in models
    assert "zhipu:glm-4.6v-flash" in models
    assert "zhipu:glm-4.7-flash" in models
    # decompose/prosecutor/defender 共用同一模型，只探一次
    assert models.count("zhipu:glm-4-flash") == 1


def test_candidates_include_judge_strong_as_judge_base():
    rows = candidates(_settings(model_judge_strong="zhipu:glm-4.5-flash"))
    entry = next(r for r in rows if r[0] == "judge_strong")
    assert entry[1] == "judge"  # 客户端按 judge 基座发起调用
    assert entry[2] == "zhipu:glm-4.5-flash"


def test_candidates_include_configured_fallbacks():
    rows = candidates(
        _settings(llm_role_fallbacks={"judge": ["siliconflow:Qwen/Qwen3-8B"]})
    )
    assert ("judge", "judge", "siliconflow:Qwen/Qwen3-8B") in rows


def test_candidates_skip_invalid_judge_strong():
    rows = candidates(_settings(model_judge_strong="not-a-provider:model"))
    assert not any(r[0] == "judge_strong" for r in rows)
