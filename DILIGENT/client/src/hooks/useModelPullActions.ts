import { useCallback } from "react";

import { pullModels } from "../services/api";

type LoadModelConfigOptions = {
  readonly syncDraft?: boolean;
};

interface UseModelPullActionsProps {
  readonly setPulling: (isPulling: boolean) => void;
  readonly setStatusMessage: (message: string) => void;
  readonly loadModelConfig: (options?: LoadModelConfigOptions) => Promise<void>;
}

interface UseModelPullActionsResult {
  readonly pullModelByName: (requestedModelName: string) => Promise<void>;
  readonly installRequiredModels: (modelNames: readonly string[]) => Promise<void>;
}

export function useModelPullActions({
  setPulling,
  setStatusMessage,
  loadModelConfig,
}: UseModelPullActionsProps): UseModelPullActionsResult {
  const runPull = useCallback(
    async (models: readonly string[], startMessage: string): Promise<void> => {
      if (!models.length) {
        return;
      }

      setPulling(true);
      setStatusMessage(startMessage);
      try {
        const result = await pullModels([...models]);
        setStatusMessage(result.message);
        await loadModelConfig({ syncDraft: false });
      } finally {
        setPulling(false);
      }
    },
    [loadModelConfig, setPulling, setStatusMessage],
  );

  const pullModelByName = useCallback(
    async (requestedModelName: string): Promise<void> => {
      const candidate = requestedModelName.trim();
      if (!candidate) {
        setStatusMessage("[ERROR] Enter a model name to pull from Ollama.");
        return;
      }
      await runPull([candidate], `[INFO] Pulling '${candidate}' from Ollama...`);
    },
    [runPull, setStatusMessage],
  );

  const installRequiredModels = useCallback(
    async (modelNames: readonly string[]): Promise<void> => {
      if (!modelNames.length) {
        return;
      }
      await runPull(
        modelNames,
        `[INFO] Installing required models: ${modelNames.join(", ")}.`,
      );
    },
    [runPull],
  );

  return {
    pullModelByName,
    installRequiredModels,
  };
}
