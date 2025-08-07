using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

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

        public static SKBitmap PreprocessSourceImage(SKBitmap sourceImage)
        {
            int RequiredWidth  = 640;
            int RequiredHeight = 640;
            var (w, h) = (sourceImage.Width, sourceImage.Height); // image width and height
            var (xRatio, yRatio) = (RequiredWidth / (float)w, RequiredHeight / (float)h); // x, y ratios
            var ratio = Math.Min(xRatio, yRatio); // ratio = resized / original
            var (width, height) = ((int)(w * ratio), (int)(h * ratio)); // roi width and height
            var (x, y) = ((RequiredWidth / 2) - (width / 2), (RequiredHeight / 2) - (height / 2)); // roi x and y coordinates

            var roi = SKRectI.Create(new SKPointI(x, y), new SKSizeI(width, height));

            SKBitmap graph = new SKBitmap(RequiredWidth, RequiredHeight);
            using SKCanvas canvas = new SKCanvas(graph);
            canvas.DrawBitmap(sourceImage, roi);

            return graph;
        }

        /// <summary>
        /// 优化原有方法
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public static Tensor<float> GetTensorForSKImage(SKBitmap image)
        {
            var bytes = image.GetPixelSpan();
            var expectedOutputLength = image.Width * image.Height * 3;
            float[] channelData = new float[expectedOutputLength];

            var expectedChannelLength = expectedOutputLength / 3;
            var greenOffset = expectedChannelLength;
            var blueOffset = expectedChannelLength * 2;

            for (int i = 0, i2 = 0; i < bytes.Length; i += 4, i2++)
            {
                var b = Convert.ToSingle(bytes[i]);
                var g = Convert.ToSingle(bytes[i + 1]);
                var r = Convert.ToSingle(bytes[i + 2]);
                channelData[i2] = (r) / 255.0f;
                channelData[i2 + greenOffset] = (g) / 255.0f;
                channelData[i2 + blueOffset] = (b) / 255.0f;
            }

            return new DenseTensor<float>(new Memory<float>(channelData), new[] { 1, 3, image.Height, image.Width });
        }

        /// <summary>
        /// Resize image with letterbox (keeping aspect ratio and padding)
        /// This is the proper way to resize images for YOLO models
        /// </summary>
        /// <param name="image">Source image</param>
        /// <param name="targetWidth">Target width</param>
        /// <param name="targetHeight">Target height</param>
        /// <returns>Resized image with letterbox padding</returns>
        public static SKBitmap ResizeImage(SKBitmap image, int targetWidth, int targetHeight)
        {
            var sourceWidth = image.Width;
            var sourceHeight = image.Height;
            
            // Calculate the scaling factor while maintaining aspect ratio
            var scaleWidth = (float)targetWidth / sourceWidth;
            var scaleHeight = (float)targetHeight / sourceHeight;
            var scale = Math.Min(scaleWidth, scaleHeight);
            
            // Calculate new dimensions
            var newWidth = (int)(sourceWidth * scale);
            var newHeight = (int)(sourceHeight * scale);
            
            // Calculate padding to center the image
            var padX = (targetWidth - newWidth) / 2;
            var padY = (targetHeight - newHeight) / 2;
            
            // Create result bitmap
            var result = new SKBitmap(targetWidth, targetHeight, image.ColorType, image.AlphaType);
            using (var canvas = new SKCanvas(result))
            {
                // Fill with gray color (0.5 * 255 = 127.5 ≈ 128) - standard letterbox color
                canvas.Clear(new SKColor(128, 128, 128));
                
                var paint = new SKPaint
                {
                    FilterQuality = SKFilterQuality.High,
                    IsAntialias = true
                };
                
                // Draw the resized image centered
                var destRect = SKRect.Create(padX, padY, newWidth, newHeight);
                canvas.DrawBitmap(image, destRect, paint);
            }
            
            return result;
        }

        public static float Clamp(float value, float min, float max)
        {
            return value < min ? min : value > max ? max : value;
        }
    }
}
