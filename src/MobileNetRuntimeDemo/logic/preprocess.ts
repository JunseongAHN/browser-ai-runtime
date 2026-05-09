import * as ort from 'onnxruntime-web';
import { MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH } from './inferenceModel';

export async function preprocess(file: File): Promise<ort.Tensor> {
  const bitmap = await createImageBitmap(file);

  const modelInputCanvas = document.createElement('canvas');
  modelInputCanvas.width = MODEL_INPUT_WIDTH;
  modelInputCanvas.height = MODEL_INPUT_HEIGHT;

  const ctx = modelInputCanvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to create canvas context');
  }

  ctx.drawImage(bitmap, 0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

  const imageData = ctx.getImageData(
    0,
    0,
    MODEL_INPUT_WIDTH,
    MODEL_INPUT_HEIGHT
  ).data;

  const pixelCount = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT;
  const tensorData = new Float32Array(3 * pixelCount);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < pixelCount; i++) {
    const r = imageData[i * 4] / 255;
    const g = imageData[i * 4 + 1] / 255;
    const b = imageData[i * 4 + 2] / 255;

    tensorData[i] = (r - mean[0]) / std[0];
    tensorData[pixelCount + i] = (g - mean[1]) / std[1];
    tensorData[pixelCount * 2 + i] = (b - mean[2]) / std[2];
  }

  return new ort.Tensor('float32', tensorData, [
    1,
    3,
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
  ]);
}
