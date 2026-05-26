import { LocalModelCard, RuntimeSettings } from '../../core/models/types';
import { resolveCloudChoices, resolveCloudModel, resolveProvider } from '../../core/model-config';
import { DraftRuntimeConfig, ModelFilterKey } from './model-config.types';

export type ModelFilterOption = {
  key: ModelFilterKey;
  label: string;
};

export const MODEL_FILTERS: readonly ModelFilterOption[] = [
  { key: 'installed', label: 'Installed in Ollama' },
  { key: 'missing', label: 'Not installed' },
  { key: 'small', label: 'Up to 8B' },
  { key: 'large', label: 'Above 8B' },
  { key: 'quantized', label: 'Quantized (Q*)' },
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

export function isSmallModel(model: LocalModelCard): boolean {
  const size = parseModelSizeInBillions(model.name);
  return size !== null && size <= 8;
}

export function isLargeModel(model: LocalModelCard): boolean {
  const size = parseModelSizeInBillions(model.name);
  return size !== null && size > 8;
}

export function isQuantizedModel(model: LocalModelCard): boolean {
  return /(?:^|[-_:])q\d/i.test(model.name);
}

export function modelMatchesFilters(
  model: LocalModelCard,
  query: string,
  filters: Record<ModelFilterKey, boolean>,
): boolean {
  const haystack = `${model.name} ${model.family} ${model.description}`.toLowerCase();
  if (query && !haystack.includes(query)) return false;
  if (filters.installed && !model.available_in_ollama) return false;
  if (filters.missing && model.available_in_ollama) return false;
  if (filters.small && !isSmallModel(model)) return false;
  if (filters.large && !isLargeModel(model)) return false;
  if (filters.quantized && !isQuantizedModel(model)) return false;
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
