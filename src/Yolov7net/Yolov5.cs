using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing;
using Yolov7net.Extentions;
using Yolov7net.Models;

namespace Yolov7net
{
    /// <summary>
    /// yolov5、yolov6 模型,不包含nms结果
    /// </summary>
    public class Yolov5 : IDisposable
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

        public List<YoloPrediction> Predict(Image image, float conf_thres = 0, float iou_thres = 0)
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

            return Suppress(ParseOutput(Inference(image), image));
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

                    var intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Area(); // intersection area
                    float unionArea = rect1.Area() + rect2.Area() - intArea; // union area
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

        private DenseTensor<float>[] Inference(Image img)
        {
            Bitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height); // fit image size to specified input size
            }
            else
            {
                resized = new Bitmap(img);
            }

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized))
            };

            var result = _inferenceSession.Run(inputs); // run inference

            var output = new List<DenseTensor<float>>(_model.Outputs.Length);

            foreach (var item in _model.Outputs) // add outputs for processing
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            }

            return output.ToArray();
        }

        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            return _model.UseDetect ? ParseDetect(output[0], image) : ParseSigmoid(output, image);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();

            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2); // left, right pads

            Parallel.For(0, (int)output.Length / _model.Dimensions, i =>
            {
                var span = output.Buffer.Span.Slice(i * _model.Dimensions);
                if (span[4] <= _model.Confidence) return; // skip low obj_conf results

                for (int j = 5; j < _model.Dimensions; j++)
                {
                    span[j] *= span[4]; // mul_conf = obj_conf * cls_conf
                }

                float xMin = (span[0] - span[2] / 2 - xPad) / gain; // unpad bbox tlx to original
                float yMin = (span[1] - span[3] / 2 - yPad) / gain; // unpad bbox tly to original
                float xMax = (span[0] + span[2] / 2 - xPad) / gain; // unpad bbox brx to original
                float yMax = (span[1] + span[3] / 2 - yPad) / gain; // unpad bbox bry to original

                xMin = Utils.Clamp(xMin, 0, w - 0); // clip bbox tlx to boundaries
                yMin = Utils.Clamp(yMin, 0, h - 0); // clip bbox tly to boundaries
                xMax = Utils.Clamp(xMax, 0, w - 1); // clip bbox brx to boundaries
                yMax = Utils.Clamp(yMax, 0, h - 1); // clip bbox bry to boundaries

                for (int k = 5; k < _model.Dimensions; k++)
                {
                    if (span[k] <= _model.MulConfidence) continue; // skip low mul_conf results

                    var label = _model.Labels[k - 5];
                    var prediction = new YoloPrediction(label, span[k])
                    {
                        Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                    };

                    result.Add(prediction);
                }
            });

            return result.ToList();
        }

        private List<YoloPrediction> ParseSigmoid(DenseTensor<float>[] output, Image image)
        {
            return new List<YoloPrediction>();
        }

        private void prepare_input(Image img)
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
