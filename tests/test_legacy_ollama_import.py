from isynkgr.llm.ollama import OllamaClient as NewOllamaClient
from isynkgr.llm_integration.ollama_client import OllamaClient as LegacyOllamaClient


def test_legacy_import_path_points_to_shared_client() -> None:
    assert LegacyOllamaClient is NewOllamaClient
