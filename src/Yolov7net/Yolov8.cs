using System.Collections.Concurrent;
using Yolov7net.Models;
using Yolov7net.Extentions;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using SkiaSharp;

namespace Yolov7net
{
    public class Yolov8 : IYoloNet
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new YoloModel();

        public Yolov8(string modelPath, bool useCuda = false)
        {

            if (useCuda)
            {
                SessionOptions opts = SessionOptions.MakeSessionOptionWithCudaProvider();
                _inferenceSession = new InferenceSession(modelPath, opts);
            }
            else
            {
                SessionOptions opts = new();
                _inferenceSession = new InferenceSession(modelPath, opts);
            }

            // Get model info
            get_input_details();
            get_output_details();
        }

        public void SetupLabels(string[] labels)
        {
            labels.Select((s, i) => new { i, s }).ToList().ForEach(item =>
            {
                _model.Labels.Add(new YoloLabel { Id = item.i, Name = item.s });
            });
        }

        public void SetupYoloDefaultLabels()
        {
            var s = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
            SetupLabels(s);
        }

        public List<YoloPrediction> Predict(SKBitmap image, float conf_thres = 0, float iou_thres = 0)
        {
            if (conf_thres > 0f)
            {
                _model.Confidence = conf_thres;
                _model.MulConfidence = conf_thres + 0.05f;
            }

            if (iou_thres > 0f)
            {
                _model.Overlap = iou_thres;
            }

            using var outputs = Inference(image);
            return Suppress(ParseOutput(outputs, image));
        }

        /// <summary>
        /// Removes overlapped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Suppress(List<YoloPrediction> items)
        {
            var areas = items.ToDictionary(item => item, item => Area(item.Rectangle));
            var toRemove = new ConcurrentBag<YoloPrediction>();

            Parallel.ForEach(items, item =>
            {
                foreach (var current in items)
                {
                    if (current == item || toRemove.Contains(current)) continue;

                    SKRect intersection;
                    intersection = SKRect.Intersect(item.Rectangle, current.Rectangle);
                    if (intersection.IsEmpty) continue;

                    float intArea = Area(intersection);
                    float unionArea = areas[item] + areas[current] - intArea;
                    float overlap = intArea / unionArea;

                    if (overlap >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            toRemove.Add(current);
                        }
                        else
                        {
                            toRemove.Add(item);
                            break;
                        }
                    }
                }
            });

            var result = items.Except(toRemove).ToList();
            return result;
        }

        // Helper method to calculate the area of a rectangle
        private float Area(SKRect rect)
        {
            return rect.Width * rect.Height;
        }


        // 使用 SkiaSharp 调整图像大小
        // 使用 SkiaSharp 调整图像大小
        // 使用 SkiaSharp 调整图像大小
        private SKBitmap ResizeImage(SKBitmap original, int width, int height)
        {
            SKImageInfo resizeInfo = new SKImageInfo(width, height);
            SKBitmap resized = new SKBitmap(resizeInfo);
            using (SKCanvas canvas = new SKCanvas(resized))
            {
                // 创建 SKPaint 对象来设置绘制的质量
                SKPaint paint = new SKPaint
                {
                    FilterQuality = SKFilterQuality.High  // 设置高质量滤镜
                };

                // 创建一个目标矩形
                SKRect destRect = new SKRect(0, 0, width, height);

                // 使用 SKPaint 对象绘制位图
                canvas.DrawBitmap(original, destRect, paint);
            }
            return resized;
        }


        // 使用 SkiaSharp 进行图像预处理
        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(SKBitmap img)
        {
            SKBitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height);
            }
            else
            {
                resized = img;
            }

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.GetTensorForSKImage(resized))
            };

            return _inferenceSession.Run(inputs, _model.Outputs);
        }


        
        private List<YoloPrediction> ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, SKBitmap image)
        {
            string firstOutput = _model.Outputs[0];
            var output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;

            return ParseDetect(output, image);
        }
        

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, SKBitmap image)
        {
            var predictions = new List<YoloPrediction>(); // 使用List收集所有预测结果
            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h);
            var gain = Math.Min(xGain, yGain);
            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2);
            var dim = output.Strides[1];

            Parallel.For(0, output.Dimensions[0], i =>
            {
                var localPredictions = new List<YoloPrediction>(); // 局部集合存储当前线程的预测
                var span = output.Buffer.Span.Slice(i * output.Strides[0]);

                for (int j = 0; j < (int)(output.Length / output.Dimensions[1]); j++)
                {
                    float a = span[j];
                    float b = span[dim + j];
                    float c = span[2 * dim + j];
                    float d = span[3 * dim + j];

                    // 预计算并重用这些值
                    float xMin = ((a - c / 2) - xPad) / gain;
                    float yMin = ((b - d / 2) - yPad) / gain;
                    float xMax = ((a + c / 2) - xPad) / gain;
                    float yMax = ((b + d / 2) - yPad) / gain;

                    for (int l = 0; l < _model.Dimensions - 4; l++)
                    {
                        var pred = span[(4 + l) * dim + j];
                        if (pred < _model.Confidence) continue;

                        var label = _model.Labels[l];
                        localPredictions.Add(new YoloPrediction
                        {
                            Label = label,
                            Score = pred,
                            Rectangle = new SKRect(xMin, yMin, xMax, yMax)
                        });
                    }
                }
                // 合并当前线程的预测结果到全局列表
                lock (predictions)
                {
                    predictions.AddRange(localPredictions);
                }
            });

            return predictions;
        }


        private void get_input_details()
        {
            _model.Height = _inferenceSession.InputMetadata["images"].Dimensions[2];
            _model.Width = _inferenceSession.InputMetadata["images"].Dimensions[3];
        }

        private void get_output_details()
        {
            _model.Outputs = _inferenceSession.OutputMetadata.Keys.ToArray();
            _model.Dimensions = _inferenceSession.OutputMetadata[_model.Outputs[0]].Dimensions[1];
            _model.UseDetect = !(_model.Outputs.Any(x => x == "score"));
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}
