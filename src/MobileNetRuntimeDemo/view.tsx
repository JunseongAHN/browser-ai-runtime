import { useEffect, useRef, useState } from 'react';
import {
  MODEL_INPUT_HEIGHT,
  MODEL_INPUT_WIDTH,
  benchmarkInferenceOnly,
} from './logic/inferenceModel';
import { useMobileNetRuntime } from './logic/useMobileNetRuntime';

type BenchmarkResult = {
  backend: string;
  runs: number;
  durationMs: number;
  fps: number;
  avgInferenceMs: number;
  minInferenceMs: number;
  maxInferenceMs: number;
};

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section style={{ border: '1px solid #ddd', padding: 16, borderRadius: 8 }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {children}
    </section>
  );
}

export default function View() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const { runtime, setBackend, setDebugImageUrl, predictFile, resetInput } =
    useMobileNetRuntime();

  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>(
    []
  );
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState('');
  const [benchmarkSampleUrl] = useState(
    `${import.meta.env.BASE_URL}models/benchmark-sample.jpg`
  );

  const cropRatio = 0.42;

  useEffect(() => {
    return () => stopCamera();
  }, []);

  async function startCamera() {
    stopCamera();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false,
    });

    streamRef.current = stream;

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    }

    resetInput();
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  async function canvasToFile(canvas: HTMLCanvasElement) {
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((result) => {
        if (result) resolve(result);
        else reject(new Error('Failed to create image blob'));
      }, 'image/jpeg');
    });

    return new File([blob], 'camera.jpg', { type: 'image/jpeg' });
  }

  async function captureRedBoxRegionAsFile() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) throw new Error('Camera not ready');

    const cropSize = Math.min(video.videoWidth, video.videoHeight) * cropRatio;
    const sx = (video.videoWidth - cropSize) / 2;
    const sy = (video.videoHeight - cropSize) / 2;

    const canvas = document.createElement('canvas');
    canvas.width = MODEL_INPUT_WIDTH;
    canvas.height = MODEL_INPUT_HEIGHT;

    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to create canvas context');

    ctx.drawImage(
      video,
      sx,
      sy,
      cropSize,
      cropSize,
      0,
      0,
      MODEL_INPUT_WIDTH,
      MODEL_INPUT_HEIGHT
    );

    setDebugImageUrl(canvas.toDataURL('image/jpeg', 0.95));
    return canvasToFile(canvas);
  }

  async function loadBenchmarkSampleFile() {
    const response = await fetch(benchmarkSampleUrl);
    if (!response.ok) throw new Error('Failed to load benchmark sample image');

    const blob = await response.blob();
    return new File([blob], 'benchmark-sample.jpg', {
      type: blob.type || 'image/jpeg',
    });
  }

  async function handleCameraPredict() {
    const file = await captureRedBoxRegionAsFile();
    await predictFile(file);
  }

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    setDebugImageUrl(imageUrl);

    await predictFile(file, imageUrl);
  }

  async function runBenchmarkForBackend(backend: 'wasm' | 'webgpu') {
    setBackend(backend);
    setBenchmarkProgress(`Loading ${backend.toUpperCase()} model...`);

    await new Promise((resolve) => setTimeout(resolve, 700));

    setBenchmarkProgress(`Running ${backend.toUpperCase()} benchmark...`);

    const file = await loadBenchmarkSampleFile();
    return benchmarkInferenceOnly(file, 10_000);
  }

  async function handleBenchmarkComparison() {
    setIsBenchmarking(true);
    setBenchmarkResults([]);

    try {
      const wasmResult = await runBenchmarkForBackend('wasm');
      setBenchmarkResults([wasmResult]);

      const webgpuResult = await runBenchmarkForBackend('webgpu');
      setBenchmarkResults([wasmResult, webgpuResult]);

      setBenchmarkProgress('Benchmark complete');
    } finally {
      setIsBenchmarking(false);
    }
  }

  return (
    <div
      style={{
        maxWidth: 1100,
        margin: '0 auto',
        padding: 24,
        fontFamily: 'system-ui',
      }}
    >
      <a href="/">← Back</a>

      <header style={{ textAlign: 'center', marginBottom: 24 }}>
        <h1>Browser AI Runtime MVP</h1>
        <p>
          Upload image, live camera inference, and fixed-sample WASM/WebGPU FPS
          comparison using ONNX Runtime Web.
        </p>
        <Panel title="This shows">
          <ul
            style={{
              paddingLeft: 20,
              margin: 0,
              textAlign: 'left',
              lineHeight: 1.8,
            }}
          >
            <li>Browser-side AI inference using ONNX Runtime Web</li>

            <li>Runtime backend switching (WASM / WebGPU)</li>

            <li>
              Image preprocessing, tensor conversion, and model input debugging
            </li>

            <li>Runtime benchmarking with FPS and latency comparison</li>

            <li>Deployable AI systems for real-world interaction</li>
          </ul>
        </Panel>
        <div>
          Runtime:{' '}
          <select
            value={runtime.backend}
            onChange={(e) => setBackend(e.target.value as 'wasm' | 'webgpu')}
          >
            <option value="wasm">WASM / CPU</option>
            <option value="webgpu">WebGPU / GPU</option>
          </select>{' '}
          {runtime.isModelReady
            ? `Ready (${runtime.backend.toUpperCase()})`
            : 'Loading model...'}
        </div>
      </header>

      {runtime.error && <p style={{ color: 'red' }}>{runtime.error}</p>}

      <div
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}
      >
        <Panel title="Option 1: Upload Image">
          <p>Classify a local image.</p>

          <button onClick={() => fileInputRef.current?.click()}>
            Choose Image
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleUpload}
            hidden
          />

          {runtime.imageUrl && (
            <img
              src={runtime.imageUrl}
              alt="Uploaded"
              style={{ width: '100%', marginTop: 12 }}
            />
          )}
        </Panel>

        <Panel title="Option 2: Live Camera">
          <p>Place object inside the red box.</p>
          <button onClick={startCamera}>Start Camera</button>{' '}
          <button
            onClick={handleCameraPredict}
            disabled={runtime.isRunning || !runtime.isModelReady}
          >
            Predict
          </button>
          <div
            style={{
              position: 'relative',
              width: '100%',
              aspectRatio: '16 / 9',
              background: '#eee',
              marginTop: 12,
              overflow: 'hidden',
            }}
          >
            <video
              ref={videoRef}
              muted
              playsInline
              autoPlay
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />

            <div
              style={{
                position: 'absolute',
                left: '50%',
                top: '50%',
                width: `${cropRatio * 100}%`,
                aspectRatio: '1 / 1',
                transform: 'translate(-50%, -50%)',
                border: '3px solid red',
                boxSizing: 'border-box',
              }}
            />
          </div>
        </Panel>

        <Panel title="Option 3: Benchmark">
          <p>Compare WASM and WebGPU using the same fixed sample image.</p>

          <img
            src={benchmarkSampleUrl}
            alt="Benchmark sample"
            style={{
              width: 140,
              height: 140,
              objectFit: 'cover',
              display: 'block',
              margin: '12px auto',
            }}
          />

          <div
            style={{
              fontSize: 14,
              lineHeight: 1.6,
              background: '#f7f7f7',
              padding: 12,
              borderRadius: 8,
              marginBottom: 12,
            }}
          >
            <strong>Benchmark setup</strong>

            <div style={{ marginTop: 6 }}>• Fixed sample image</div>

            <div>
              • {MODEL_INPUT_WIDTH}×{MODEL_INPUT_HEIGHT} input
            </div>

            <div>• 10 second inference-only test</div>

            <div>• Camera capture excluded</div>
          </div>

          <button
            onClick={handleBenchmarkComparison}
            disabled={isBenchmarking || !runtime.isModelReady}
          >
            {isBenchmarking ? 'Benchmarking...' : 'Compare WASM vs WebGPU'}
          </button>

          {benchmarkResults.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <strong>Result</strong>

              <table style={{ marginTop: 8, width: '100%' }}>
                <thead>
                  <tr>
                    <th align="left">Backend</th>
                    <th align="right">FPS</th>
                    <th align="right">Avg ms</th>
                  </tr>
                </thead>

                <tbody>
                  {benchmarkResults.map((result) => (
                    <tr key={result.backend}>
                      <td>{result.backend.toUpperCase()}</td>
                      <td align="right">{result.fps.toFixed(2)}</td>
                      <td align="right">{result.avgInferenceMs.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {benchmarkResults.length === 2 && (
                <p>
                  WebGPU is{' '}
                  <strong>
                    {(
                      benchmarkResults[1].fps / benchmarkResults[0].fps
                    ).toFixed(2)}
                    ×
                  </strong>
                  faster than WASM on this sample.
                </p>
              )}
            </div>
          )}

          {benchmarkProgress && (
            <p
              style={{
                marginTop: 10,
                color: '#666',
                fontSize: 14,
              }}
            >
              {benchmarkProgress}
            </p>
          )}
        </Panel>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 16,
          marginTop: 16,
        }}
      >
        <Panel title="Model Input">
          {runtime.debugImageUrl ? (
            <img
              src={runtime.debugImageUrl}
              alt="Model input"
              style={{ width: 224, height: 224, border: '1px solid #ddd' }}
            />
          ) : (
            <p>No model input yet.</p>
          )}

          <p>
            Input tensor: {MODEL_INPUT_WIDTH}×{MODEL_INPUT_HEIGHT}
          </p>
        </Panel>

        <Panel title="Model Output">
          {runtime.topPrediction ? (
            <>
              <h2>{runtime.topPrediction.label}</h2>
              <p>
                Class {runtime.topPrediction.index} ·{' '}
                {(runtime.topPrediction.score * 100).toFixed(2)}%
              </p>

              {runtime.topPredictions.map((p) => (
                <div key={p.index} style={{ marginBottom: 8 }}>
                  <div>
                    {p.label} — {(p.score * 100).toFixed(2)}%
                  </div>
                  <div style={{ height: 8, background: '#eee' }}>
                    <div
                      style={{
                        height: '100%',
                        width: `${Math.min(p.score * 100, 100)}%`,
                        background: '#111',
                      }}
                    />
                  </div>
                </div>
              ))}
            </>
          ) : (
            <p>No prediction yet.</p>
          )}
        </Panel>
      </div>
    </div>
  );
}
