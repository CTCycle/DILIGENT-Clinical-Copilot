import React from "react";

export type StatusTone = "is-error" | "is-info" | "is-success";

interface StatusMessageProps {
  readonly message: string;
  readonly tone?: StatusTone;
  readonly className?: string;
}

export function resolveStatusTone(message: string): StatusTone {
  const normalized = message.trim().toUpperCase();
  if (!normalized) {
    return "is-info";
  }
  if (normalized.startsWith("[ERROR]")) {
    return "is-error";
  }
  if (normalized.startsWith("[INFO]")) {
    return "is-info";
  }
  return "is-success";
}

export function StatusMessage({
  message,
  tone,
  className = "model-config-status-message",
}: StatusMessageProps): React.JSX.Element | null {
  const normalized = message.trim();
  if (!normalized) {
    return null;
  }

  const resolvedTone = tone ?? resolveStatusTone(normalized);
  return (
    <p
      className={`${className} ${resolvedTone}`}
      role={resolvedTone === "is-error" ? "alert" : "status"}
      aria-live={resolvedTone === "is-error" ? "assertive" : "polite"}
    >
      {normalized}
    </p>
  );
}
