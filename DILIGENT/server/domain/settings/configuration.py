from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FastAPISettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    title: str
    description: str
    version: str


class JobsSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    polling_interval: float


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    embedded_database: bool
    engine: str | None
    host: str | None
    port: int | None
    database_name: str | None
    username: str | None
    password: str | None
    ssl: bool
    ssl_ca: str | None
    connect_timeout: int
    insert_batch_size: int
    insert_commit_interval: int
    select_page_size: int


class DrugsMatcherSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    direct_confidence: float
    master_confidence: float
    synonym_confidence: float
    partial_confidence: float
    fuzzy_confidence: float
    fuzzy_threshold: float
    token_max_frequency: int
    token_min_length: int
    normalization_cache_limit: int
    variant_cache_limit: int
    match_cache_limit: int
    alias_cache_limit: int
    min_confidence: float
    catalog_excluded_term_suffixes: tuple[str, ...]
    catalog_token_ratio_threshold: float
    catalog_overall_ratio_threshold: float
    fuzzy_early_exit_ratio: float
    catalog_index_limit: int
    catalog_candidate_limit: int


class RagSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    vector_collection_name: str
    chunk_size: int
    chunk_overlap: int
    embedding_batch_size: int
    use_reranking: bool
    rerank_candidate_k: int
    rerank_top_n: int
    embedding_backend: str
    ollama_base_url: str
    ollama_embedding_model: str
    hf_embedding_model: str
    vector_index_metric: str
    vector_index_type: str
    reset_vector_collection: bool
    cloud_provider: str
    cloud_model: str
    cloud_embedding_model: str
    use_cloud_embeddings: bool
    vector_stream_batch_size: int
    embedding_max_workers: int


class ExternalDataSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    default_llm_timeout: float
    parser_llm_timeout: float
    disease_llm_timeout: float
    clinical_llm_timeout: float
    livertox_llm_timeout: float
    ollama_server_start_timeout: float
    livertox_archive: str
    livertox_yield_interval: int
    livertox_skip_deterministic_ratio: float
    livertox_monograph_max_workers: int
    max_excerpt_length: int
    rxnav_request_timeout: float
    rxnav_max_concurrency: int
    tavily_request_timeout_s: float
    tavily_search_cache_ttl_s: int
    tavily_extract_cache_ttl_s: int
    tavily_rate_limit_per_minute: int
    tavily_fast_max_results: int
    tavily_thorough_max_results: int
    tavily_extract_top_urls: int


class IngestionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    drug_name_min_length: int
    drug_name_max_length: int
    drug_name_max_tokens: int


class LLMRuntimeDefaults(BaseModel):
    model_config = ConfigDict(frozen=True)
    parsing_model: str
    clinical_model: str
    llm_provider: str
    cloud_model: str
    use_cloud_services: bool
    ollama_temperature: float
    cloud_temperature: float
    ollama_reasoning: bool
    ollama_host_default: str


class ServerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    fastapi: FastAPISettings
    jobs: JobsSettings
    database: DatabaseSettings
    drugs_matcher: DrugsMatcherSettings
    rag: RagSettings
    external_data: ExternalDataSettings
    ingestion: IngestionSettings
    llm_defaults: LLMRuntimeDefaults
