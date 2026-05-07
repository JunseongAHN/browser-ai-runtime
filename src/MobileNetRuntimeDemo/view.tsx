import { useEffect, useRef, useState, type ReactNode } from 'react';

import { useMobileNetRuntime } from './logic/useMobileNetRuntime';
import type { Backend } from './logic/inferenceModel';

const pageStyle = {
  maxWidth: 1040,
  margin: '40px auto',
  padding: 24,
  fontFamily: 'Arial, sans-serif',
  color: '#111',
} as const;

const gridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
  gap: 12,
} as const;

export default function MobileNetRuntimeDemoView() {
  const runtime = useMobileNetRuntime();

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const loopRef = useRef<number | null>(null);
  const runningRef = useRef(false);

  const [cameraStatus, setCameraStatus] = useState('Camera idle');
  const [streamMode, setStreamMode] = useState<'camera' | 'debug' | null>(null);
  const [streamFrames, setStreamFrames] = useState(0);
  const [streamStartedAt, setStreamStartedAt] = useState<number | null>(null);

  const streamFps =
    streamStartedAt !== null
      ? streamFrames / Math.max((performance.now() - streamStartedAt) / 1000, 1)
      : null;

  useEffect(() => {
    return () => {
      stopStreamLoop();
      stopCamera();
    };
  }, []);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraStatus('Camera ready');
      return true;
    } catch (error) {
      console.error(error);
      setCameraStatus(
        'Camera unavailable. Use Debug Loop with uploaded image.'
      );
      return false;
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraStatus('Camera stopped');
  }

  function stopStreamLoop() {
    runningRef.current = false;

    if (loopRef.current !== null) {
      window.clearTimeout(loopRef.current);
      loopRef.current = null;
    }

    setStreamMode(null);
  }

  async function captureVideoFrameAsFile() {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (
      !video ||
      !canvas ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return null;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');
    if (!context) return null;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, 'image/jpeg', 0.9);
    });

    if (!blob) return null;

    return new File([blob], 'webcam-frame.jpg', {
      type: 'image/jpeg',
    });
  }

  async function startCameraLoop() {
    if (!runtime.modelLoaded) return;

    const cameraReady = streamRef.current ? true : await startCamera();
    if (!cameraReady) return;

    runningRef.current = true;
    setStreamMode('camera');
    setStreamFrames(0);
    setStreamStartedAt(performance.now());
    setCameraStatus('Camera stream running');

    async function tick() {
      if (!runningRef.current) return;

      const frame = await captureVideoFrameAsFile();

      if (frame) {
        await runtime.runStreamFrame(frame);
        setStreamFrames((previous) => previous + 1);
      }

      loopRef.current = window.setTimeout(tick, 250);
    }

    tick();
  }

  async function startDebugLoop() {
    if (!runtime.modelLoaded || !runtime.hasImage) return;

    runningRef.current = true;
    setStreamMode('debug');
    setStreamFrames(0);
    setStreamStartedAt(performance.now());
    setCameraStatus('Debug loop running with uploaded image');

    async function tick() {
      if (!runningRef.current) return;

      await runtime.runAgain();
      setStreamFrames((previous) => previous + 1);

      loopRef.current = window.setTimeout(tick, 250);
    }

    tick();
  }

  return (
    <main style={pageStyle}>
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ marginBottom: 24 }}>ONNX Edge Runtime Benchmark</h1>

        <p style={{ color: '#555', maxWidth: 760, lineHeight: 1.5 }}>
          Browser-based ONNX inference benchmark for lightweight edge AI models
          using WASM and WebGPU backends.
        </p>
      </header>

      <Panel title="Runtime Controls">
        <div
          style={{
            display: 'flex',
            gap: 12,
            alignItems: 'center',
            flexWrap: 'wrap',
          }}
        >
          <label>
            Backend{' '}
            <select
              value={runtime.backend}
              onChange={(event) =>
                runtime.setBackend(event.target.value as Backend)
              }
            >
              <option value="wasm">WASM</option>
              <option value="webgpu">WebGPU</option>
            </select>
          </label>

          <button onClick={runtime.loadModel}>Load Model</button>

          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              runtime.uploadAndRun(event.target.files?.[0]);
            }}
          />

          <button
            onClick={runtime.runAgain}
            disabled={!runtime.modelLoaded || !runtime.hasImage}
          >
            Run Again
          </button>

          <button
            onClick={() => runtime.runBenchmark(10)}
            disabled={!runtime.modelLoaded || !runtime.hasImage}
          >
            Run 10x Benchmark
          </button>

          <button
            onClick={runtime.clearResults}
            disabled={runtime.runCount === 0}
          >
            Clear
          </button>
        </div>

        <p style={{ marginBottom: 0, color: '#666' }}>
          Status: {runtime.status}
        </p>
      </Panel>

      <Panel title="Real-Time Mode">
        <div
          style={{
            display: 'flex',
            gap: 12,
            alignItems: 'center',
            flexWrap: 'wrap',
          }}
        >
          <button
            onClick={startCameraLoop}
            disabled={!runtime.modelLoaded || streamMode !== null}
          >
            Start Camera Loop
          </button>

          <button
            onClick={startDebugLoop}
            disabled={
              !runtime.modelLoaded || !runtime.hasImage || streamMode !== null
            }
          >
            Start Debug Loop
          </button>

          <button onClick={stopStreamLoop} disabled={streamMode === null}>
            Stop Loop
          </button>

          <button onClick={stopCamera}>Stop Camera</button>
        </div>

        <p style={{ marginBottom: 0, color: '#666' }}>
          {cameraStatus}
          {streamMode ? ` · Mode: ${streamMode}` : ''}
        </p>
      </Panel>

      <section style={{ ...gridStyle, marginTop: 16, marginBottom: 16 }}>
        <MetricCard label="Backend" value={runtime.backend.toUpperCase()} />
        <MetricCard
          label="Session"
          value={runtime.modelLoaded ? 'READY' : 'NOT LOADED'}
        />
        <MetricCard
          label="Model Load"
          value={
            runtime.loadTimeMs !== null
              ? `${runtime.loadTimeMs.toFixed(2)} ms`
              : '-'
          }
        />
        <MetricCard
          label="Last"
          value={
            runtime.lastLatency !== null
              ? `${runtime.lastLatency.toFixed(2)} ms`
              : '-'
          }
        />
        <MetricCard
          label="Avg"
          value={
            runtime.averageLatency !== null
              ? `${runtime.averageLatency.toFixed(2)} ms`
              : '-'
          }
        />
        <MetricCard
          label="Min"
          value={
            runtime.minLatency !== null
              ? `${runtime.minLatency.toFixed(2)} ms`
              : '-'
          }
        />
        <MetricCard
          label="Max"
          value={
            runtime.maxLatency !== null
              ? `${runtime.maxLatency.toFixed(2)} ms`
              : '-'
          }
        />
        <MetricCard label="Runs" value={runtime.runCount.toString()} />
        <MetricCard
          label="Stream FPS"
          value={streamFps !== null ? streamFps.toFixed(2) : '-'}
        />
      </section>

      {runtime.benchmarkSummary && (
        <Panel title="Benchmark Summary">
          <div style={gridStyle}>
            <MetricCard
              label="Backend"
              value={runtime.benchmarkSummary.backend.toUpperCase()}
            />
            <MetricCard
              label="Iterations"
              value={runtime.benchmarkSummary.iterations.toString()}
            />
            <MetricCard
              label="Min"
              value={`${runtime.benchmarkSummary.minLatency.toFixed(2)} ms`}
            />
            <MetricCard
              label="Avg"
              value={`${runtime.benchmarkSummary.avgLatency.toFixed(2)} ms`}
            />
            <MetricCard
              label="Max"
              value={`${runtime.benchmarkSummary.maxLatency.toFixed(2)} ms`}
            />
            <MetricCard
              label="Time"
              value={runtime.benchmarkSummary.timestamp}
            />
          </div>
        </Panel>
      )}

      <section
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: 16,
          marginTop: 16,
        }}
      >
        <Panel title="Prediction">
          {runtime.topPrediction ? (
            <>
              <div style={{ marginBottom: 16 }}>
                <div style={{ color: '#666', fontSize: 13 }}>
                  Top Prediction
                </div>

                <div style={{ fontSize: 24, fontWeight: 700 }}>
                  {runtime.topPrediction.label}
                </div>

                <div style={{ color: '#666', marginTop: 4 }}>
                  Class {runtime.topPrediction.index} ·{' '}
                  {(runtime.topPrediction.score * 100).toFixed(2)}%
                </div>
              </div>

              <div>
                <div style={{ color: '#666', fontSize: 13, marginBottom: 8 }}>
                  Top-5 Predictions
                </div>

                <ol style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
                  {runtime.topPredictions.map((prediction) => (
                    <li key={prediction.index}>
                      {prediction.label}{' '}
                      <span style={{ color: '#666' }}>
                        ({(prediction.score * 100).toFixed(2)}%)
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            </>
          ) : (
            <p style={{ color: '#666' }}>No prediction yet.</p>
          )}
        </Panel>

        <Panel title={streamMode === 'camera' ? 'Camera Input' : 'Input Image'}>
          {streamMode === 'camera' ? (
            <video
              ref={videoRef}
              muted
              playsInline
              style={{
                width: '100%',
                maxHeight: 360,
                objectFit: 'contain',
                borderRadius: 8,
                background: '#f6f6f6',
              }}
            />
          ) : runtime.imageUrl ? (
            <img
              src={runtime.imageUrl}
              alt="Uploaded"
              style={{
                width: '100%',
                maxHeight: 360,
                objectFit: 'contain',
                borderRadius: 8,
                background: '#f6f6f6',
              }}
            />
          ) : (
            <p style={{ color: '#666' }}>No image uploaded yet.</p>
          )}
        </Panel>
      </section>

      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </main>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        border: '1px solid #e5e5e5',
        borderRadius: 10,
        padding: 14,
        background: '#fff',
      }}
    >
      <div style={{ fontSize: 12, color: '#777', marginBottom: 6 }}>
        {label}
      </div>

      <div style={{ fontSize: 18, fontWeight: 700, wordBreak: 'break-word' }}>
        {value}
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section
      style={{
        border: '1px solid #e5e5e5',
        borderRadius: 12,
        padding: 18,
        background: '#fff',
        marginBottom: 16,
      }}
    >
      <h2 style={{ fontSize: 18, marginTop: 0, marginBottom: 16 }}>{title}</h2>
      {children}
    </section>
  );
}
