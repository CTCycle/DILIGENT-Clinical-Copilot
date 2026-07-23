from __future__ import annotations

from dataclasses import replace

from common.embedding.config import (
    CANONICAL_EMBEDDING_CONFIG,
    CanonicalEmbeddingConfig,
    canonical_embedding_fingerprint,
)


def test_canonical_config_is_the_granite_contract() -> None:
    config = CANONICAL_EMBEDDING_CONFIG

    assert config.model_id == "ibm-granite/granite-embedding-small-english-r2"
    assert config.revision == "2ab6fa8ea2d674564defd37171ae19079b864b33"
    assert config.dimension == 384
    assert config.pooling == "cls"
    assert config.normalize is True
    assert config.trust_remote_code is False


def test_fingerprint_is_deterministic() -> None:
    assert canonical_embedding_fingerprint() == CANONICAL_EMBEDDING_CONFIG.fingerprint
    assert canonical_embedding_fingerprint() == canonical_embedding_fingerprint()


def test_fingerprint_changes_for_every_semantic_field() -> None:
    config = CANONICAL_EMBEDDING_CONFIG
    fields = (
        "schema_version",
        "model_id",
        "revision",
        "dimension",
        "pooling",
        "normalize",
        "query_prefix",
        "document_prefix",
        "distance_metric",
        "maximum_model_tokens",
        "maximum_query_tokens",
        "default_chunk_tokens",
        "default_chunk_overlap_tokens",
        "output_dtype",
        "trust_remote_code",
    )

    for field in fields:
        value = getattr(config, field)
        if isinstance(value, bool):
            changed = not value
        elif isinstance(value, int):
            changed = value + 1
        else:
            changed = f"{value}-changed"
        assert replace(config, **{field: changed}).fingerprint != config.fingerprint


def test_config_is_frozen() -> None:
    config = CanonicalEmbeddingConfig(**CANONICAL_EMBEDDING_CONFIG.to_canonical_dict())

    try:
        config.dimension = 768  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("canonical embedding configuration must be immutable")
