export type Prediction = {
  index: number;
  label: string;
  score: number;
};

export function postprocess(
  output: Float32Array,
  labels: string[],
  topK = 5
): {
  topPrediction: Prediction;
  topPredictions: Prediction[];
} {
  const probabilities = softmax(Array.from(output));

  const topPredictions = probabilities
    .map((score, index) => ({
      index,
      label: labels[index] ?? `Class ${index}`,
      score,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  return {
    topPrediction: topPredictions[0],
    topPredictions,
  };
}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);

  return exps.map((x) => x / sum);
}
