import React, { useEffect, useMemo, useState } from "react";

import { InspectionUpdateJobState } from "../hooks/useInspectionUpdateJob";
import { InspectionUpdateConfigResponse } from "../types";
import { InspectionUpdateConfigStep } from "./InspectionUpdateConfigStep";
import { InspectionUpdateConfirmStep } from "./InspectionUpdateConfirmStep";
import { InspectionUpdateProgressStep } from "./InspectionUpdateProgressStep";

type InspectionUpdateWizardProps = {
  targetLabel: string;
  fallbackMessage: string;
  loadConfig: () => Promise<InspectionUpdateConfigResponse>;
  startJob: (payload?: Record<string, unknown>) => Promise<void>;
  job: InspectionUpdateJobState;
};

type WizardStep = 1 | 2 | 3;

export function InspectionUpdateWizard({
  targetLabel,
  fallbackMessage,
  loadConfig,
  startJob,
  job,
}: InspectionUpdateWizardProps): React.JSX.Element {
  const [step, setStep] = useState<WizardStep>(1);
  const [configError, setConfigError] = useState<string | null>(null);
  const [allowedFields, setAllowedFields] = useState<string[]>([]);
  const [values, setValues] = useState<Record<string, unknown>>({});

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      try {
        const payload = await loadConfig();
        if (cancelled) {
          return;
        }
        setAllowedFields(payload.allowed_fields);
        setValues(payload.defaults);
        setConfigError(null);
      } catch (error) {
        if (cancelled) {
          return;
        }
        setConfigError(error instanceof Error ? error.message : "Failed to load configuration.");
      }
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [loadConfig]);

  useEffect(() => {
    if (job.running || job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
      setStep(3);
    }
  }, [job.running, job.status]);

  const filteredPayload = useMemo<Record<string, unknown>>(() => {
    const allowed = new Set(allowedFields);
    return Object.fromEntries(Object.entries(values).filter(([key]) => allowed.has(key)));
  }, [allowedFields, values]);

  const handleChange = (field: string, value: unknown): void => {
    setValues((previous) => ({ ...previous, [field]: value }));
  };

  const start = async (): Promise<void> => {
    await startJob(filteredPayload);
    setStep(3);
  };

  const cancel = async (): Promise<void> => {
    await startJob();
  };

  if (configError) {
    return <p className="inspection-error-text">{configError}</p>;
  }

  if (step === 1) {
    return (
      <InspectionUpdateConfigStep
        values={values}
        allowedFields={allowedFields}
        disabled={job.running}
        onChange={handleChange}
        onNext={() => setStep(2)}
      />
    );
  }

  if (step === 2) {
    return (
      <InspectionUpdateConfirmStep
        targetLabel={targetLabel}
        values={filteredPayload}
        disabled={job.running}
        onBack={() => setStep(1)}
        onStart={() => {
          void start();
        }}
      />
    );
  }

  return (
    <InspectionUpdateProgressStep
      job={job}
      fallbackMessage={fallbackMessage}
      onCancel={() => {
        void cancel();
      }}
      onReset={() => setStep(1)}
    />
  );
}

