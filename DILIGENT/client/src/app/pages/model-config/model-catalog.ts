import { LocalModelCard, RuntimeSettings } from '../../core/models/types';
import { resolveCloudChoices, resolveCloudModel, resolveProvider } from '../../core/model-config';
import { DraftRuntimeConfig, ModelFilterKey } from './model-config.types';

export type ModelFilterOption = {
  key: ModelFilterKey;
  label: string;
};

export const MODEL_FILTERS: readonly ModelFilterOption[] = [
  { key: 'installed', label: 'Installed' },
  { key: 'reasoning', label: 'Reasoning' },
  { key: 'small', label: 'Small models' },
  { key: 'extraction', label: 'Extraction' },
];

export function parseModelSizeInBillions(name: string): number | null {
  const match = name.match(/:(\d+(?:\.\d+)?)([mb])$/i);
  if (!match) {
    return null;
  }
  const value = Number.parseFloat(match[1]);
  if (!Number.isFinite(value)) {
    return null;
  }
  return match[2].toLowerCase() === 'm' ? value / 1000 : value;
}

export function isReasoningModel(model: LocalModelCard): boolean {
  const value = `${model.name} ${model.family} ${model.description}`.toLowerCase();
  return value.includes('reasoning');
}

export function isSmallModel(model: LocalModelCard): boolean {
  const size = parseModelSizeInBillions(model.name);
  return size !== null && size <= 4;
}

export function isExtractionModel(model: LocalModelCard): boolean {
  const value = `${model.name} ${model.family} ${model.description}`.toLowerCase();
  const extractionKeywords = [
    'extract',
    'parsing',
    'parser',
    'structured',
    'compact',
    'lightweight',
    'low-latency',
    'smollm',
  ];
  return extractionKeywords.some((keyword) => value.includes(keyword)) || isSmallModel(model);
}

export function modelMatchesFilters(
  model: LocalModelCard,
  query: string,
  filters: Record<ModelFilterKey, boolean>,
): boolean {
  const haystack = `${model.name} ${model.family} ${model.description}`.toLowerCase();
  if (query && !haystack.includes(query)) return false;
  if (filters.installed && !model.available_in_ollama) return false;
  if (filters.reasoning && !isReasoningModel(model)) return false;
  if (filters.small && !isSmallModel(model)) return false;
  if (filters.extraction && !isExtractionModel(model)) return false;
  return true;
}

export function resolveDraftFromSettings(runtimeSettings: RuntimeSettings): DraftRuntimeConfig {
  const choices = resolveCloudChoices(undefined);
  const provider = resolveProvider(runtimeSettings.provider, choices);
  const cloudModel = resolveCloudModel(provider, runtimeSettings.cloudModel, choices);
  return {
    useCloudServices: runtimeSettings.useCloudServices,
    provider,
    cloudModel,
    clinicalModel: runtimeSettings.clinicalModel,
    textExtractionModel: runtimeSettings.textExtractionModel,
    temperature: runtimeSettings.temperature,
  };
}
