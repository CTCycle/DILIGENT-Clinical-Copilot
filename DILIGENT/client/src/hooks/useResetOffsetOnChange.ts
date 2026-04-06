import { Dispatch, SetStateAction, useEffect } from "react";

export function useResetOffsetOnChange(
  setOffset: Dispatch<SetStateAction<number>>,
  dependencies: readonly unknown[],
): void {
  useEffect(() => {
    setOffset(0);
  }, [setOffset, ...dependencies]);
}
