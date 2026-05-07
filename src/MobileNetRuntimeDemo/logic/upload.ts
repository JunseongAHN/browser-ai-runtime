export interface UploadedImage {
  file: File;
  imageUrl: string;
}

export function uploadImage(file: File): UploadedImage {
  if (!file.type.startsWith("image/")) {
    throw new Error("Uploaded file must be an image");
  }

  return {
    file,
    imageUrl: URL.createObjectURL(file),
  };
}