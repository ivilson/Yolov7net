using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace Yolov7net.Extentions
{
    public static class Utils
    {
        /// <summary>
        /// xywh to xyxy
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        public static SKBitmap ResizeImage(SKBitmap image, int targetWidth, int targetHeight)
        {
            var resized = new SKBitmap(targetWidth, targetHeight, image.ColorType, image.AlphaType);
            using (var canvas = new SKCanvas(resized))
            {
                canvas.Clear(SKColors.Transparent);
                var paint = new SKPaint
                {
                    FilterQuality = SKFilterQuality.High,
                    IsAntialias = true
                };
                canvas.DrawBitmap(image, SKRect.Create(0, 0, targetWidth, targetHeight), paint);
            }
            return resized;
        }

        public static Tensor<float> ExtractPixels(SKBitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

            using (var pixmap = bitmap.PeekPixels())
            {
                var pixels = pixmap.GetPixelSpan<byte>();
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int idx = (y * width + x) * 4; // Assuming 4 bytes per pixel (RGBA)
                        tensor[0, 0, y, x] = pixels[idx + 2] / 255.0f; // R
                        tensor[0, 1, y, x] = pixels[idx + 1] / 255.0f; // G
                        tensor[0, 2, y, x] = pixels[idx] / 255.0f;     // B
                    }
                }
            }
            return tensor;
        }

        //https://github.com/ivilson/Yolov7net/issues/17
        public static Tensor<float> ExtractPixels2(SKBitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

            using (var pixmap = bitmap.PeekPixels())
            {
                var pixels = pixmap.GetPixelSpan<byte>();
                int bytesPerPixel = pixmap.BytesPerPixel;
                int pixelCount = width * height;

                for (int i = 0; i < pixelCount; i++)
                {
                    int idx = i * bytesPerPixel;
                    float r = pixels[idx + 2] / 255.0f;  // Assuming RGBA or BGRA
                    float g = pixels[idx + 1] / 255.0f;
                    float b = pixels[idx] / 255.0f;

                    if (pixmap.ColorType == SKColorType.Rgba8888 || pixmap.ColorType == SKColorType.Bgra8888)
                    {
                        // Adjust index based on color type
                        if (pixmap.ColorType == SKColorType.Bgra8888)
                        {
                            r = pixels[idx] / 255.0f;
                            g = pixels[idx + 1] / 255.0f;
                            b = pixels[idx + 2] / 255.0f;
                        }

                        tensor[0, 0, i / width, i % width] = r;
                        tensor[0, 1, i / width, i % width] = g;
                        tensor[0, 2, i / width, i % width] = b;
                    }
                }
            }

            return tensor;
        }



        public static float Clamp(float value, float min, float max)
        {
            return value < min ? min : value > max ? max : value;
        }
    }
}
