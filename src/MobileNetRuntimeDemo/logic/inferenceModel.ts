import * as ort from 'onnxruntime-web';
import { preprocess } from './preprocess';
import { parseImagenetClasses } from './parseImagenetClasses';
import { postprocess, type Prediction } from './postprocess';

export type { Prediction };
export const MODEL_INPUT_WIDTH = 224;
export const MODEL_INPUT_HEIGHT = 224;

export type Backend = 'wasm' | 'webgpu';

export type InferenceResult = {
  topPrediction: Prediction;
  topPredictions: Prediction[];
};

let session: ort.InferenceSession | null = null;
let labels: string[] | null = null;
let currentBackend: Backend | null = null;

export async function initializeModel(backend: Backend): Promise<void> {
  if (session && currentBackend === backend) return;

  currentBackend = backend;

  const baseUrl = import.meta.env.BASE_URL;

  session = await ort.InferenceSession.create(
    `${baseUrl}models/mobilenetv2-7.onnx`,
    {
      executionProviders: backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
    }
  );

  labels = await parseImagenetClasses(`${baseUrl}models/imagenet_classes.txt`);
}

export async function runInference(file: File): Promise<InferenceResult> {
  if (!session) throw new Error('Model is not initialized');
  if (!labels) throw new Error('ImageNet labels are not loaded');

  const input = await preprocess(file);

  const outputs = await session.run({
    [session.inputNames[0]]: input,
  });

  const output = outputs[session.outputNames[0]].data as Float32Array;

  return postprocess(output, labels);
}

export async function benchmarkInferenceOnly(file: File, durationMs = 10_000) {
  if (!session) throw new Error('Model is not initialized');

  const input = await preprocess(file);
  const inputName = session.inputNames[0];

  const times: number[] = [];
  const start = performance.now();

  while (performance.now() - start < durationMs) {
    const t0 = performance.now();

    await session.run({
      [inputName]: input,
    });

    times.push(performance.now() - t0);
  }

  const totalMs = performance.now() - start;
  const sum = times.reduce((a, b) => a + b, 0);

  return {
    backend: currentBackend ?? 'unknown',
    runs: times.length,
    durationMs: totalMs,
    fps: times.length / (totalMs / 1000),
    avgInferenceMs: sum / times.length,
    minInferenceMs: Math.min(...times),
    maxInferenceMs: Math.max(...times),
  };
}
