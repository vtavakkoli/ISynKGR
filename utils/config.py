"""
Configuration settings for the extended hybrid framework.

This module contains configuration parameters for all components of the framework,
including paths, model settings, and hyperparameters.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR.parent / "data"
MODELS_DIR = BASE_DIR.parent / "models"
OUTPUT_DIR = BASE_DIR.parent / "output"
LOGS_DIR = BASE_DIR.parent / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Standards configuration
STANDARDS = {
    "IEEE_1451": {
        "name": "IEEE 1451",
        "description": "Standard for smart sensor and actuator integration",
        "files": [str(DATA_DIR / "standards" / "ieee_1451.txt")],
    },
    "ISO_15926": {
        "name": "ISO 15926",
        "description": "Standard for process industries data integration",
        "files": [str(DATA_DIR / "standards" / "iso_15926.txt")],
    },
    "IEC_61499": {
        "name": "IEC 61499",
        "description": "Standard for distributed industrial process measurement and control systems",
        "files": [str(DATA_DIR / "standards" / "iec_61499.txt")],
    },
}

# Ontology Learning configuration
ONTOLOGY_LEARNING = {
    "preprocessing": {
        "min_token_length": 3,
        "max_token_length": 50,
        "stopwords_removal": True,
        "lemmatization": True,
    },
    "concept_extraction": {
        "min_concept_frequency": 5,
        "max_concepts": 1000,
        "similarity_threshold": 0.75,
    },
    "relation_extraction": {
        "min_relation_confidence": 0.6,
        "max_relations_per_concept": 20,
    },
    "validation": {
        "consistency_check": True,
        "completeness_threshold": 0.8,
    },
}

# Knowledge Graph configuration
KNOWLEDGE_GRAPH = {
    "database": {
        "type": "neo4j",
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password",
    },
    "entity_extraction": {
        "confidence_threshold": 0.7,
        "max_entities": 5000,
    },
    "relationship_extraction": {
        "confidence_threshold": 0.65,
        "max_relationships": 10000,
    },
    "community_detection": {
        "algorithm": "leiden",
        "resolution_parameter": 1.0,
        "min_community_size": 3,
    },
    "graph_embeddings": {
        "dimension": 128,
        "p": 1,
        "q": 1,
        "num_walks": 10,
        "walk_length": 80,
    },
}

# LLM configuration
LLM = {
    "models": {
        "gpt4o": {
            "name": "gpt-4o",
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        "gemini": {
            "name": "gemini-2.0-pro",
            "provider": "google",
            "api_key_env": "GOOGLE_API_KEY",
            "max_tokens": 8192,
            "temperature": 0.8,
        },
        "claude": {
            "name": "claude-3.5-sonnet",
            "provider": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        "deepseek": {
            "name": "deepseek-r1",
            "provider": "deepseek",
            "api_key_env": "DEEPSEEK_API_KEY",
            "max_tokens": 4096,
            "temperature": 0.6,
        },
    },
    "default_model": "gpt4o",
    "prompting": {
        "use_chain_of_thought": True,
        "use_few_shot": True,
        "num_examples": 3,
    },
}

# Hybrid Framework configuration
HYBRID_FRAMEWORK = {
    "integration": {
        "use_graphrag": True,
        "use_parallel_retrievers": True,
        "max_retrievers": 3,
    },
    "pipeline": {
        "batch_size": 32,
        "num_workers": 4,
        "cache_results": True,
    },
    "orchestration": {
        "timeout": 300,  # seconds
        "retry_attempts": 3,
        "error_threshold": 0.1,
    },
}

# Evaluation configuration
EVALUATION = {
    "metrics": {
        "precision": True,
        "recall": True,
        "f1_score": True,
        "accuracy": True,
    },
    "benchmarking": {
        "baseline_comparison": True,
        "ablation_study": True,
        "cross_validation": True,
        "num_folds": 5,
    },
    "performance": {
        "track_memory": True,
        "track_time": True,
        "track_api_calls": True,
    },
}

# Logging configuration
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "framework.log"),
}
