from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[2]


def _load_config_module():
    spec = spec_from_file_location("google_search_plugin_config_under_test", PLUGIN_ROOT / "config.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载插件配置模块")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_module = _load_config_module()


def test_tavily_subagent_is_opt_in_and_bounded() -> None:
    config = config_module.GoogleSearchPluginConfig()

    assert config.tavily_subagent.enabled is False
    assert config.tavily_subagent.max_rounds == 4
    assert config.tavily_subagent.max_retries == 2
    assert config.tavily_subagent.extract_depth == "basic"
    assert config.tavily_subagent.extract_chunks_per_source == 3


def test_legacy_tavily_content_options_are_preserved() -> None:
    config = config_module.GoogleSearchPluginConfig(
        engines={
            "tavily_include_answer": True,
            "tavily_include_raw_content": True,
        }
    )

    assert config.engines.tavily_include_answer is True
    assert config.engines.tavily_include_raw_content is True
