export async function parseImagenetClasses(path: string): Promise<string[]> {
  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(`Failed to load ImageNet classes from ${path}`);
  }

  const text = await response.text();

  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}
