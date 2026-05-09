export interface UploadedImage {
  file: File;
  imageUrl: string;
}

export function uploadImage(file: File): UploadedImage {
  if (!file.type.startsWith('image/')) {
    throw new Error('Uploaded file must be an image');
  }

  return {
    file,
    imageUrl: URL.createObjectURL(file),
  };
}

export async function loadBenchmarkSampleFile() {
  const baseUrl = import.meta.env.BASE_URL;

  const response = await fetch(`${baseUrl}models/benchmark-sample.jpg`);
  if (!response.ok) {
    throw new Error('Failed to load benchmark sample image');
  }

  const blob = await response.blob();

  return new File([blob], 'benchmark-sample.jpg', {
    type: blob.type || 'image/jpeg',
  });
}
