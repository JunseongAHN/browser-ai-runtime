import * as ort from "onnxruntime-web";

export type Backend = "wasm" | "webgpu";

export interface CreateMobileNetModelOptions {
  modelPath: string;
  labelPath: string;
  backend: Backend;
  inputWidth?: number;
  inputHeight?: number;
}

export interface Prediction {
  index: number;
  label: string;
  score: number;
}

export interface MobileNetInferenceResult {
  latencyMs: number;
  topPrediction: Prediction;
  topPredictions: Prediction[];
  rawOutput: ort.Tensor;
}

export interface MobileNetModel {
  backend: Backend;
  loadTimeMs: number;
  session: ort.InferenceSession;
  inference: (file: File) => Promise<MobileNetInferenceResult>;
}

async function loadLabels(labelPath: string): Promise<string[]> {
  const response = await fetch(labelPath);

  if (!response.ok) {
    throw new Error(`Failed to load labels from ${labelPath}`);
  }

  const text = await response.text();

  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export async function createMobileNetModel({
  modelPath,
  labelPath,
  backend,
  inputWidth = 224,
  inputHeight = 224,
}: CreateMobileNetModelOptions): Promise<MobileNetModel> {
  const loadStart = performance.now();


  const executionProviders =
  backend === "webgpu"
    ? ["webgpu", "wasm"]
    : ["wasm"];

  const [session, labels] = await Promise.all([
    ort.InferenceSession.create(modelPath, {
      executionProviders: executionProviders,
    }),
    loadLabels(labelPath),
  ]);

  const loadTimeMs = performance.now() - loadStart;

  async function preprocess(file: File): Promise<ort.Tensor> {
    const bitmap = await createImageBitmap(file);

    const canvas = document.createElement("canvas");
    canvas.width = inputWidth;
    canvas.height = inputHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Failed to create canvas context");
    }

    ctx.drawImage(bitmap, 0, 0, inputWidth, inputHeight);

    const imageData = ctx.getImageData(0, 0, inputWidth, inputHeight).data;
    const floatData = new Float32Array(1 * 3 * inputWidth * inputHeight);

    for (let i = 0; i < inputWidth * inputHeight; i++) {
      const r = imageData[i * 4] / 255;
      const g = imageData[i * 4 + 1] / 255;
      const b = imageData[i * 4 + 2] / 255;

      floatData[i] = r;
      floatData[inputWidth * inputHeight + i] = g;
      floatData[2 * inputWidth * inputHeight + i] = b;
    }

    return new ort.Tensor("float32", floatData, [
      1,
      3,
      inputHeight,
      inputWidth,
    ]);
  }

  async function forward(inputTensor: ort.Tensor): Promise<ort.Tensor> {
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const outputs = await session.run({
      [inputName]: inputTensor,
    });

    const output = outputs[outputName];

    if (!(output instanceof ort.Tensor)) {
      throw new Error("Model output is not an ONNX tensor");
    }

    return output;
  }

  function softmax(values: Float32Array): number[] {
    const maxValue = Math.max(...values);
    const exps = Array.from(values, (value) => Math.exp(value - maxValue));
    const sum = exps.reduce((acc, value) => acc + value, 0);

    return exps.map((value) => value / sum);
  }

  function postprocess(output: ort.Tensor): {
    topPrediction: Prediction;
    topPredictions: Prediction[];
  } {
    const data = output.data;

    if (!(data instanceof Float32Array)) {
      throw new Error("Expected Float32Array output");
    }

    const probabilities = softmax(data);

    const topPredictions = probabilities
      .map((score, index) => ({
        index,
        label: labels[index] ?? `Class ${index}`,
        score,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    return {
      topPrediction: topPredictions[0],
      topPredictions,
    };
  }

  async function inference(file: File): Promise<MobileNetInferenceResult> {
    const inputTensor = await preprocess(file);

    const start = performance.now();
    const output = await forward(inputTensor);
    const latencyMs = performance.now() - start;

    const { topPrediction, topPredictions } = postprocess(output);

    return {
      latencyMs,
      topPrediction,
      topPredictions,
      rawOutput: output,
    };
  }

  return {
    backend,
    loadTimeMs,
    session,
    inference,
  };
}