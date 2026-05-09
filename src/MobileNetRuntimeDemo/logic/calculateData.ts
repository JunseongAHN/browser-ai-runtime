export type BenchmarkResult = {
  frames: number;
  durationMs: number;
  fps: number;
  avgInferenceMs: number;
  minInferenceMs: number;
  maxInferenceMs: number;
};

export async function runTimedBenchmark({
  durationMs,
  captureFrame,
  runFrame,
}: {
  durationMs: number;
  captureFrame: () => Promise<File>;
  runFrame: (file: File) => Promise<void>;
}): Promise<BenchmarkResult> {
  const start = performance.now();
  const times: number[] = [];

  while (performance.now() - start < durationMs) {
    const file = await captureFrame();

    const inferenceStart = performance.now();
    await runFrame(file);
    const inferenceEnd = performance.now();

    times.push(inferenceEnd - inferenceStart);

    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  const end = performance.now();
  const totalMs = end - start;
  const frames = times.length;

  const sum = times.reduce((a, b) => a + b, 0);

  return {
    frames,
    durationMs: totalMs,
    fps: frames / (totalMs / 1000),
    avgInferenceMs: sum / frames,
    minInferenceMs: Math.min(...times),
    maxInferenceMs: Math.max(...times),
  };
}
