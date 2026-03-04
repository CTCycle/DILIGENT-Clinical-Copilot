import { useCallback, useEffect, useMemo, useState } from "react";

import {
  activateAccessKey,
  createAccessKey,
  deleteAccessKey,
  fetchAccessKeys,
} from "../services/api";
import { AccessKeyProvider, AccessKeyRecord } from "../types";

export interface AccessKeyManagerState {
  keys: AccessKeyRecord[];
  sortedKeys: AccessKeyRecord[];
  isLoading: boolean;
  isSaving: boolean;
  newKeyValue: string;
  errorMessage: string;
  visibleRows: Record<number, boolean>;
  hasKeys: boolean;
}

export interface AccessKeyManagerActions {
  setNewKeyValue: (value: string) => void;
  handleAdd: () => Promise<void>;
  handleActivate: (keyId: number) => Promise<void>;
  handleDelete: (keyId: number) => Promise<void>;
  toggleVisibility: (keyId: number) => void;
}

export interface AccessKeyManagerResult {
  state: AccessKeyManagerState;
  actions: AccessKeyManagerActions;
}

export function useAccessKeyManager(
  provider: AccessKeyProvider,
  isOpen: boolean,
): AccessKeyManagerResult {
  const [keys, setKeys] = useState<AccessKeyRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [newKeyValue, setNewKeyValue] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [visibleRows, setVisibleRows] = useState<Record<number, boolean>>({});

  const sortedKeys = useMemo(
    () =>
      [...keys].sort(
        (left, right) =>
          Number(right.is_active) - Number(left.is_active) || right.id - left.id,
      ),
    [keys],
  );

  const loadKeys = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    setErrorMessage("");
    try {
      const payload = await fetchAccessKeys(provider);
      setKeys(payload);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unable to load access keys.";
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  }, [provider]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    void loadKeys();
  }, [isOpen, loadKeys]);

  const handleAdd = useCallback(async (): Promise<void> => {
    const candidate = newKeyValue.trim();
    if (!candidate) {
      setErrorMessage("Please paste a key before adding.");
      return;
    }
    setIsSaving(true);
    setErrorMessage("");
    try {
      await createAccessKey(provider, candidate);
      setNewKeyValue("");
      await loadKeys();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unable to add access key.";
      setErrorMessage(message);
    } finally {
      setIsSaving(false);
    }
  }, [loadKeys, newKeyValue, provider]);

  const handleActivate = useCallback(async (keyId: number): Promise<void> => {
    setIsSaving(true);
    setErrorMessage("");
    try {
      const activated = await activateAccessKey(keyId, provider);
      setKeys((current) =>
        current.map((item) => ({
          ...item,
          is_active: item.id === activated.id,
          updated_at: item.id === activated.id ? activated.updated_at : item.updated_at,
          last_used_at:
            item.id === activated.id ? activated.last_used_at : item.last_used_at,
        })),
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unable to activate access key.";
      setErrorMessage(message);
    } finally {
      setIsSaving(false);
    }
  }, [provider]);

  const handleDelete = useCallback(async (keyId: number): Promise<void> => {
    setIsSaving(true);
    setErrorMessage("");
    try {
      await deleteAccessKey(keyId, provider);
      setKeys((current) => current.filter((item) => item.id !== keyId));
      setVisibleRows((current) => {
        const next = { ...current };
        delete next[keyId];
        return next;
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unable to delete access key.";
      setErrorMessage(message);
    } finally {
      setIsSaving(false);
    }
  }, [provider]);

  const toggleVisibility = useCallback((keyId: number): void => {
    setVisibleRows((current) => ({ ...current, [keyId]: !current[keyId] }));
  }, []);

  return {
    state: {
      keys,
      sortedKeys,
      isLoading,
      isSaving,
      newKeyValue,
      errorMessage,
      visibleRows,
      hasKeys: keys.length > 0,
    },
    actions: {
      setNewKeyValue,
      handleAdd,
      handleActivate,
      handleDelete,
      toggleVisibility,
    },
  };
}
