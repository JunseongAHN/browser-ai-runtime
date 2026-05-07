import { useState } from 'react';

import {
  createMobileNetModel,
  type Backend,
  type MobileNetModel,
  type Prediction,
} from './inferenceModel';

import { uploadImage } from './upload';

const MODEL_PATH = '/models/mobilenetv2-7.onnx';
const LABEL_PATH = '/models/imagenet_classes.txt';

type BenchmarkSummary = {
  backend: Backend;
  iterations: number;
  minLatency: number;
  avgLatency: number;
  maxLatency: number;
  timestamp: string;
};

export function useMobileNetRuntime() {
  const [backend, setBackendState] = useState<Backend>('wasm');
  const [model, setModel] = useState<MobileNetModel | null>(null);

  const [status, setStatus] = useState('Idle');
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);

  const [loadTimeMs, setLoadTimeMs] = useState<number | null>(null);
  const [latencies, setLatencies] = useState<number[]>([]);

  const [benchmarkSummary, setBenchmarkSummary] =
    useState<BenchmarkSummary | null>(null);

  const [topPrediction, setTopPrediction] = useState<Prediction | null>(null);
  const [topPredictions, setTopPredictions] = useState<Prediction[]>([]);

  const lastLatency =
    latencies.length > 0 ? latencies[latencies.length - 1] : null;

  const averageLatency =
    latencies.length > 0
      ? latencies.reduce((sum, value) => sum + value, 0) / latencies.length
      : null;

  const minLatency = latencies.length > 0 ? Math.min(...latencies) : null;
  const maxLatency = latencies.length > 0 ? Math.max(...latencies) : null;

  function resetInferenceState() {
    setModel(null);
    setLoadTimeMs(null);
    setLatencies([]);
    setBenchmarkSummary(null);
    setTopPrediction(null);
    setTopPredictions([]);
  }

  function setBackend(nextBackend: Backend) {
    setBackendState(nextBackend);
    resetInferenceState();
    setStatus('Backend changed. Please reload the model.');
  }

  async function loadModel() {
    try {
      setStatus(`Loading MobileNet with ${backend.toUpperCase()}...`);

      const loadedModel = await createMobileNetModel({
        modelPath: MODEL_PATH,
        labelPath: LABEL_PATH,
        backend,
      });

      setModel(loadedModel);
      setLoadTimeMs(loadedModel.loadTimeMs);
      setLatencies([]);
      setBenchmarkSummary(null);
      setTopPrediction(null);
      setTopPredictions([]);

      setStatus(`MobileNet loaded with ${backend.toUpperCase()}`);
    } catch (error) {
      console.error(error);
      setModel(null);

      setStatus(
        backend === 'webgpu'
          ? 'Failed to load MobileNet with WebGPU. WebGPU may not be supported in this browser.'
          : 'Failed to load MobileNet with WASM.'
      );
    }
  }

  async function runInference(
    file: File,
    options?: {
      recordLatency?: boolean;
      updatePreview?: boolean;
      updateStatus?: boolean;
    }
  ) {
    if (!model) {
      setStatus('Please load the model first.');
      return null;
    }

    const recordLatency = options?.recordLatency ?? true;
    const updatePreview = options?.updatePreview ?? true;
    const updateStatus = options?.updateStatus ?? true;

    try {
      if (updatePreview) {
        const uploadedImage = uploadImage(file);
        setImageUrl(uploadedImage.imageUrl);
      }

      if (updateStatus) {
        setStatus('Running inference...');
      }

      const result = await model.inference(file);

      if (recordLatency) {
        setLatencies((previous) => [...previous, result.latencyMs]);
      }

      setTopPrediction(result.topPrediction);
      setTopPredictions(result.topPredictions);

      if (updateStatus) {
        setStatus('Inference complete');
      }

      return result;
    } catch (error) {
      console.error(error);
      setStatus('Inference failed.');
      return null;
    }
  }

  async function uploadAndRun(file?: File) {
    if (!file) return;

    setCurrentFile(file);
    await runInference(file);
  }

  async function runAgain() {
    if (!currentFile) {
      setStatus('Please upload an image first.');
      return;
    }

    await runInference(currentFile);
  }

  async function runStreamFrame(file: File) {
    return runInference(file, {
      recordLatency: true,
      updatePreview: false,
      updateStatus: false,
    });
  }

  async function runBenchmark(iterations = 10) {
    if (!currentFile) {
      setStatus('Please upload an image first.');
      return;
    }

    if (!model) {
      setStatus('Please load the model first.');
      return;
    }

    setStatus('Running warm-up inference...');

    await runInference(currentFile, {
      recordLatency: false,
      updatePreview: false,
    });

    const benchmarkLatencies: number[] = [];

    setLatencies([]);

    for (let i = 0; i < iterations; i += 1) {
      setStatus(`Running benchmark ${i + 1}/${iterations}...`);

      const result = await runInference(currentFile, {
        updatePreview: false,
      });

      if (result) {
        benchmarkLatencies.push(result.latencyMs);
      }
    }

    if (benchmarkLatencies.length === 0) {
      setStatus('Benchmark failed.');
      return;
    }

    const min = Math.min(...benchmarkLatencies);
    const max = Math.max(...benchmarkLatencies);
    const avg =
      benchmarkLatencies.reduce((sum, value) => sum + value, 0) /
      benchmarkLatencies.length;

    setBenchmarkSummary({
      backend,
      iterations,
      minLatency: min,
      avgLatency: avg,
      maxLatency: max,
      timestamp: new Date().toLocaleTimeString(),
    });

    setStatus(`${iterations}x benchmark complete`);
  }

  function clearResults() {
    setLatencies([]);
    setBenchmarkSummary(null);
    setTopPrediction(null);
    setTopPredictions([]);
    setStatus('Results cleared.');
  }

  return {
    backend,
    setBackend,

    status,
    imageUrl,

    modelLoaded: model !== null,
    hasImage: currentFile !== null,

    loadTimeMs,
    lastLatency,
    averageLatency,
    minLatency,
    maxLatency,
    runCount: latencies.length,

    benchmarkSummary,

    topPrediction,
    topPredictions,

    loadModel,
    uploadAndRun,
    runAgain,
    runStreamFrame,
    runBenchmark,
    clearResults,
  };
}
