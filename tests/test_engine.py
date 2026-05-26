"""Unit tests for UniversalRAGEngine without database dependencies."""

import pytest

from docs2db_api.rag.engine import RAGConfig
from docs2db_api.rag.engine import UniversalRAGEngine


class TestUniversalRAGEngineInit:
    def test_default_init(self):
        engine = UniversalRAGEngine()

        assert engine.config is not None
        assert engine.llm_client is None
        assert engine.db_manager is None
        assert engine.embedding_provider is None
        assert engine._started is False

    def test_custom_config(self):
        config = RAGConfig(model_name="test-model", similarity_threshold=0.9)
        engine = UniversalRAGEngine(config=config, refinement_prompt="custom {question}")

        assert engine.config.model_name == "test-model"
        assert engine.config.similarity_threshold == 0.9
        assert engine.refinement_prompt == "custom {question}"

    def test_db_config_passthrough(self):
        db_config = {"host": "localhost", "port": "5432", "dbname": "test"}
        engine = UniversalRAGEngine(db_config=db_config)

        assert engine._db_config_dict == db_config

    def test_llm_client_kwarg_rejected(self):
        with pytest.raises(TypeError, match="llm_client"):
            UniversalRAGEngine(llm_client="should_not_work")
