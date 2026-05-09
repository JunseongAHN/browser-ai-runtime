import { useCallback, useEffect, useState } from 'react';
import {
  initializeModel,
  runInference,
  type Backend,
  type Prediction,
} from './inferenceModel';

export type MobileNetRuntimeState = {
  backend: Backend;
  isModelReady: boolean;
  isRunning: boolean;
  imageUrl: string | null;
  debugImageUrl: string | null;
  topPrediction: Prediction | null;
  topPredictions: Prediction[];
  error: string | null;
};

export function useMobileNetRuntime() {
  const [runtime, setRuntime] = useState<MobileNetRuntimeState>({
    backend: 'wasm',
    isModelReady: false,
    isRunning: false,
    imageUrl: null,
    debugImageUrl: null,
    topPrediction: null,
    topPredictions: [],
    error: null,
  });

  useEffect(() => {
    let cancelled = false;

    setRuntime((prev) => ({
      ...prev,
      isModelReady: false,
      error: null,
    }));

    initializeModel(runtime.backend)
      .then(() => {
        if (cancelled) return;

        setRuntime((prev) => ({
          ...prev,
          isModelReady: true,
        }));
      })
      .catch((error) => {
        if (cancelled) return;

        setRuntime((prev) => ({
          ...prev,
          isModelReady: false,
          error:
            error instanceof Error
              ? error.message
              : 'Failed to initialize model',
        }));
      });

    return () => {
      cancelled = true;
    };
  }, [runtime.backend]);

  const setBackend = useCallback((backend: Backend) => {
    setRuntime((prev) => ({
      ...prev,
      backend,
      isModelReady: false,
      error: null,
    }));
  }, []);

  const setDebugImageUrl = useCallback((debugImageUrl: string | null) => {
    setRuntime((prev) => ({
      ...prev,
      debugImageUrl,
    }));
  }, []);

  const predictFile = useCallback(async (file: File, imageUrl?: string) => {
    try {
      setRuntime((prev) => ({
        ...prev,
        imageUrl: imageUrl ?? prev.imageUrl,
        debugImageUrl: imageUrl ?? prev.debugImageUrl,
        isRunning: true,
        error: null,
      }));

      const result = await runInference(file);

      setRuntime((prev) => ({
        ...prev,
        topPrediction: result.topPrediction,
        topPredictions: result.topPredictions,
        isRunning: false,
      }));
    } catch (error) {
      setRuntime((prev) => ({
        ...prev,
        isRunning: false,
        error:
          error instanceof Error ? error.message : 'Failed to run inference',
      }));
    }
  }, []);

  const resetInput = useCallback(() => {
    setRuntime((prev) => ({
      ...prev,
      imageUrl: null,
      debugImageUrl: null,
      topPrediction: null,
      topPredictions: [],
      error: null,
    }));
  }, []);

  return {
    runtime,
    setBackend,
    setDebugImageUrl,
    predictFile,
    resetInput,
  };
}
