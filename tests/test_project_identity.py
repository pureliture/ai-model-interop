from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_pyproject_keeps_legacy_distribution_name_and_updates_description():
    pyproject = tomllib.loads(_read("pyproject.toml"))

    assert pyproject["project"]["name"] == "openai-compatible-bridge"
    assert pyproject["project"]["description"] == "Multi-protocol gateway for AI model APIs across providers"


def test_dockerfile_uses_package_entrypoint():
    dockerfile = _read("Dockerfile")

    assert "COPY openai_compatible_bridge" in dockerfile
    assert "openai_compatible_bridge.main:app" in dockerfile
    assert "app.py" not in dockerfile
    assert "vertex.py" not in dockerfile
    assert "cost_tracking.py" not in dockerfile
    assert "ARG NEURONS_SOURCE_COMMIT" in dockerfile
    assert "ARG NEURONS_SOURCE_REPOSITORY" in dockerfile
    assert "org.opencontainers.image.revision" in dockerfile
    assert "org.opencontainers.image.source" in dockerfile


def test_compose_and_env_use_bridge_identity():
    compose = _read("docker-compose.yml")
    env_example = _read(".env.example")

    assert "openai-compatible-bridge" in compose
    assert "BRIDGE_API_KEY" in compose
    assert "BRIDGE_API_KEY" in env_example
    assert "MODEL_REGISTRY_JSON" in compose
    assert "MODEL_REGISTRY_JSON" in env_example
    assert "OLLAMA_BASE_URL" in compose
    assert "OLLAMA_BASE_URL" in env_example
    assert "OLLAMA_HTTP_TIMEOUT_SECONDS" in compose
    assert "OLLAMA_HTTP_TIMEOUT_SECONDS" in env_example
    assert "OLLAMA_THINK" in compose
    assert "OLLAMA_THINK" in env_example
    assert "FOUNDRY_BASE_URL" in compose
    assert "FOUNDRY_BASE_URL" in env_example
    assert "FOUNDRY_TOKEN" in compose
    assert "FOUNDRY_TOKEN" in env_example
    assert "wrapper-vertex-ai-api" not in compose
    assert "WRAPPER_API_KEY" not in compose
    assert "WRAPPER_API_KEY" not in env_example


def test_public_identity_uses_ai_model_interop_and_keeps_runtime_names():
    readme = _read("README.md")
    banner = _read("assets/header.svg")
    app_source = _read("openai_compatible_bridge/main.py")
    publish_workflow = _read(".github/workflows/reusable-docker-publish.yml")

    assert "AI Model Interop" in readme
    assert "AI Model Interop" in banner
    assert 'title="AI Model Interop"' in app_source
    assert 'title="openai-compatible-bridge"' not in app_source
    assert "org.opencontainers.image.title=AI Model Interop" in publish_workflow
    assert "org.opencontainers.image.description=Multi-protocol gateway for AI model APIs across providers" in publish_workflow
    assert "http://openai-compatible-bridge" in readme
    assert "vertex-ai-api-wrapper" not in readme
    assert "wrapper-vertex-ai-api" not in readme
    assert "WRAPPER_API_KEY" not in readme


def test_exported_fastapi_app_uses_new_project_title():
    from openai_compatible_bridge.main import app

    assert app.title == "AI Model Interop"
