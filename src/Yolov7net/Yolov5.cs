using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using Yolov7net.Extentions;
using Yolov7net.Models;

namespace Yolov7net
{
    /// <summary>
    /// yolov5、yolov6 模型,不包含nms结果
    /// </summary>
    public class Yolov5 : IYoloNet
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new YoloModel();

        public Yolov5(string modelPath, bool useCuda = false)
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
        /// Removes overlaped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Suppress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items) // iterate every prediction
            {
                foreach (var current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    SKRect intersection;
                    intersection = SKRect.Intersect(item.Rectangle, current.Rectangle);
                    if (intersection.IsEmpty) continue;
                    float intArea = Area(intersection); // intersection area
                    float unionArea = Area(rect1) + Area(rect2) - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    if (overlap >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result;
        }

        private float Area(SKRect rect)
        {
            return rect.Width * rect.Height;
        }

        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(SKBitmap img)
        {
            SKBitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height); // fit image size to specified input size
            }
            else
            {
                resized = img;
            }

            var inputs = new[] // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.GetTensorForSKImage(resized))
            };

            return _inferenceSession.Run(inputs, _model.Outputs); // run inference
        }

        private List<YoloPrediction> ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, SKBitmap image)
        {
            if (_model.UseDetect)
            {
                string firstOutput = _model.Outputs[0];
                var output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
                return ParseDetect(output, image);
            }

            return ParseSigmoid(outputs, image);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, SKBitmap image)
        {
            var predictions = new List<YoloPrediction>();

            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h);
            var gain = Math.Min(xGain, yGain);
            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2);

            Parallel.For(0, (int)output.Length / _model.Dimensions, i =>
            {
                var localPredictions = new List<YoloPrediction>();
                var span = output.Buffer.Span.Slice(i * _model.Dimensions);
                if (span[4] <= _model.Confidence) return;

                float xMin = Math.Max((span[0] - span[2] / 2 - xPad) / gain, 0);
                float yMin = Math.Max((span[1] - span[3] / 2 - yPad) / gain, 0);
                float xMax = Math.Min((span[0] + span[2] / 2 - xPad) / gain, w - 1);
                float yMax = Math.Min((span[1] + span[3] / 2 - yPad) / gain, h - 1);

                for (int k = 5; k < _model.Dimensions; k++)
                {
                    span[k] *= span[4]; // mul_conf = obj_conf * cls_conf
                    if (span[k] <= _model.MulConfidence) continue;

                    var label = _model.Labels[k - 5];
                    localPredictions.Add(new YoloPrediction(label, span[k])
                    {
                        Rectangle = new SKRect(xMin, yMin, xMax , yMax)
                    });
                }

                // 合并到全局结果
                lock (predictions)
                {
                    predictions.AddRange(localPredictions);
                }
            });

            return predictions;
        }


        private List<YoloPrediction> ParseSigmoid(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output, SKBitmap image)
        {
            return new List<YoloPrediction>();
        }

        private void prepare_input(SKBitmap img)
        {
            var bmp = Utils.ResizeImage(img, _model.Width, _model.Height);
        }

        private void get_input_details()
        {
            _model.Height = _inferenceSession.InputMetadata["images"].Dimensions[2];
            _model.Width = _inferenceSession.InputMetadata["images"].Dimensions[3];
        }

        private void get_output_details()
        {
            _model.Outputs = _inferenceSession.OutputMetadata.Keys.ToArray();
            _model.Dimensions = _inferenceSession.OutputMetadata[_model.Outputs[0]].Dimensions[2];
            _model.UseDetect = !(_model.Outputs.Any(x=>x == "score"));
        }

        public void Dispose()
        {
            _inferenceSession.Dispose(); 
        }
    }
}
