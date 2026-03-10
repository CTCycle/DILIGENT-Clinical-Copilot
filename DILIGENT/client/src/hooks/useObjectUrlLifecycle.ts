import { useCallback, useEffect, useRef } from "react";

interface ObjectUrlLifecycleResult {
  readonly revokeObjectUrl: (url: string | null) => void;
}

export function useObjectUrlLifecycle(
  currentUrl: string | null,
): ObjectUrlLifecycleResult {
  const objectUrlRef = useRef<string | null>(currentUrl);

  useEffect(() => {
    objectUrlRef.current = currentUrl;
  }, [currentUrl]);

  const revokeObjectUrl = useCallback((url: string | null): void => {
    if (!url) {
      return;
    }
    URL.revokeObjectURL(url);
    if (objectUrlRef.current === url) {
      objectUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  return { revokeObjectUrl };
}

