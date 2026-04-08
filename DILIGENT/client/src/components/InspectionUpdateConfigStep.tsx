import React, { useMemo } from "react";

import { InspectionUpdateTarget } from "../types";

type InspectionUpdateConfigStepProps = {
  target: InspectionUpdateTarget;
  values: Record<string, unknown>;
  allowedFields: string[];
  disabled: boolean;
  onChange: (field: string, value: unknown) => void;
  onNext: () => void;
};

type RagProvider = "ollama" | "openai" | "google";

const RAG_FIELD_ORDER = [
  "chunk_size",
  "chunk_overlap",
  "embedding_batch_size",
  "vector_stream_batch_size",
  "embedding_max_workers",
] as const;

const RAG_PROVIDER_OPTIONS: { value: RagProvider; label: string }[] = [
  { value: "ollama", label: "ollama" },
  { value: "openai", label: "openai" },
  { value: "google", label: "google" },
];

const RAG_PROVIDER_DEFAULT_MODELS: Record<RagProvider, string> = {
  ollama: "nomic-embed-text:latest",
  openai: "text-embedding-3-large",
  google: "text-embedding-004",
};

function parseRawValue(rawValue: string, current: unknown): unknown {
  if (typeof current === "number") {
    const numeric = Number.parseFloat(rawValue);
    return Number.isFinite(numeric) ? numeric : current;
  }
  if (typeof current === "boolean") {
    return rawValue === "true";
  }
  return rawValue;
}

export function humanizeInspectionFieldLabel(field: string): string {
  return field
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^./, (character) => character.toUpperCase());
}

function resolveRagProvider(values: Record<string, unknown>): RagProvider {
  const backend = typeof values.embedding_backend === "string"
    ? values.embedding_backend.trim().toLowerCase()
    : "";
  if (backend === "ollama") {
    return "ollama";
  }

  const cloudProvider = typeof values.cloud_provider === "string"
    ? values.cloud_provider.trim().toLowerCase()
    : "";
  if (cloudProvider === "gemini") {
    return "google";
  }
  if (cloudProvider === "openai") {
    return "openai";
  }

  if (values.use_cloud_embeddings === true) {
    return "openai";
  }

  return "ollama";
}

function resolveRagModelValue(values: Record<string, unknown>, provider: RagProvider): string {
  if (provider === "ollama") {
    const model = typeof values.ollama_embedding_model === "string" ? values.ollama_embedding_model.trim() : "";
    return model || RAG_PROVIDER_DEFAULT_MODELS.ollama;
  }

  const model = typeof values.cloud_embedding_model === "string" ? values.cloud_embedding_model.trim() : "";
  return model || RAG_PROVIDER_DEFAULT_MODELS[provider];
}

function applyRagProviderSelection(
  provider: RagProvider,
  onChange: (field: string, value: unknown) => void,
): void {
  if (provider === "ollama") {
    onChange("embedding_backend", "ollama");
    onChange("use_cloud_embeddings", false);
    onChange("ollama_embedding_model", RAG_PROVIDER_DEFAULT_MODELS.ollama);
    return;
  }

  onChange("embedding_backend", "cloud");
  onChange("use_cloud_embeddings", true);
  onChange("cloud_provider", provider === "google" ? "gemini" : "openai");
  onChange("cloud_embedding_model", RAG_PROVIDER_DEFAULT_MODELS[provider]);
}

export function normalizeRagUpdateDefaults(values: Record<string, unknown>): Record<string, unknown> {
  const next = { ...values };
  const provider = resolveRagProvider(next);
  if (provider === "ollama") {
    next.embedding_backend = "ollama";
    next.use_cloud_embeddings = false;
    next.ollama_embedding_model = resolveRagModelValue(next, "ollama");
    return next;
  }

  next.embedding_backend = "cloud";
  next.use_cloud_embeddings = true;
  next.cloud_provider = provider === "google" ? "gemini" : "openai";
  next.cloud_embedding_model = resolveRagModelValue(next, provider);
  return next;
}

function renderGenericField(
  field: string,
  value: unknown,
  disabled: boolean,
  onChange: (field: string, value: unknown) => void,
): React.JSX.Element | null {
  if (typeof value === "boolean") {
    return (
      <label key={field} className="field checkbox">
        <input
          type="checkbox"
          checked={value}
          onChange={(event) => onChange(field, event.target.checked)}
          disabled={disabled}
        />
        <span className="field-label">{humanizeInspectionFieldLabel(field)}</span>
      </label>
    );
  }

  return (
    <label key={field} className="field">
      <span className="field-label">{humanizeInspectionFieldLabel(field)}</span>
      <input
        type={typeof value === "number" ? "number" : "text"}
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(event) => onChange(field, parseRawValue(event.target.value, value))}
        disabled={disabled}
      />
    </label>
  );
}

function renderRagField(
  field: string,
  value: unknown,
  disabled: boolean,
  onChange: (field: string, value: unknown) => void,
): React.JSX.Element | null {
  if (field === "vector_stream_batch_size") {
    return (
      <label key={field} className="field">
        <span className="field-label">{humanizeInspectionFieldLabel(field)}</span>
        <input
          type="number"
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(event) => onChange(field, parseRawValue(event.target.value, value))}
          disabled={disabled}
        />
      </label>
    );
  }

  return (
    <label key={field} className="field">
      <span className="field-label">{humanizeInspectionFieldLabel(field)}</span>
      <input
        type={typeof value === "number" ? "number" : "text"}
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(event) => onChange(field, parseRawValue(event.target.value, value))}
        disabled={disabled}
      />
    </label>
  );
}

export function InspectionUpdateConfigStep({
  target,
  values,
  allowedFields,
  disabled,
  onChange,
  onNext,
}: InspectionUpdateConfigStepProps): React.JSX.Element {
  const isRag = target === "rag";
  const ragProvider = useMemo(() => (isRag ? resolveRagProvider(values) : "ollama"), [isRag, values]);
  const ragModelValue = useMemo(
    () => (isRag ? resolveRagModelValue(values, ragProvider) : ""),
    [isRag, ragProvider, values],
  );

  if (!isRag) {
    return (
      <div className="inspection-widget">
        <h4>Step 1 · Configuration</h4>
        <div className="inspection-controls inspection-controls-sessions">
          {allowedFields.map((field) => {
            const value = values[field];
            return renderGenericField(field, value, disabled, onChange);
          })}
        </div>
        <div className="inspection-pager-actions">
          <button
            type="button"
            className="btn btn-primary inspection-mini-btn"
            onClick={onNext}
            disabled={disabled}
          >
            Continue
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="inspection-widget inspection-update-config-step">
      <div className="inspection-update-config-grid">
        <div className="inspection-update-config-column">
          {RAG_FIELD_ORDER.map((field) => {
            if (!allowedFields.includes(field)) {
              return null;
            }
            return renderRagField(field, values[field], disabled, onChange);
          })}
        </div>
        <div className="inspection-update-config-column inspection-update-config-column-secondary">
          <label className="field">
            <span className="field-label">Provider</span>
            <select
              className="inspection-update-provider-select"
              value={ragProvider}
              onChange={(event) => applyRagProviderSelection(event.target.value as RagProvider, onChange)}
              disabled={disabled}
            >
              {RAG_PROVIDER_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <div className="field inspection-model-display">
            <span className="field-label">Embedding model</span>
            <div className="inspection-model-display-value">{ragModelValue}</div>
          </div>

          {allowedFields.includes("reset_vector_collection") && (
            <label className="field checkbox inspection-reset-checkbox">
              <input
                type="checkbox"
                checked={Boolean(values.reset_vector_collection)}
                onChange={(event) => onChange("reset_vector_collection", event.target.checked)}
                disabled={disabled}
              />
              <span className="field-label">Reset the vector store</span>
            </label>
          )}
        </div>
      </div>

      <div className="inspection-pager-actions">
        <button
          type="button"
          className="btn btn-primary inspection-mini-btn inspection-update-continue"
          onClick={onNext}
          disabled={disabled}
        >
          Continue
        </button>
      </div>
    </div>
  );
}
