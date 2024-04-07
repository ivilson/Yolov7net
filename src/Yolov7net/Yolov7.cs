using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Concurrent;
using System.Drawing;
using Yolov7net.Extentions;
using Yolov7net.Models;


namespace Yolov7net
{
    public  class Yolov7 : IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new YoloModel();

        public Yolov7(string modelPath, bool useCuda = false)
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

        public List<YoloPrediction> Predict(Image image)
        {
            using var outputs = Inference(image);
            string firstOutput = _model.Outputs[0];
            var output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
            return ParseDetect(output, image);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var predictions = new List<YoloPrediction>();

            var (w, h) = (image.Width, image.Height);
            var (xGain, yGain) = (_model.Width / (float)w, _model.Height / (float)h);
            var gain = Math.Min(xGain, yGain);
            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2);

            for (int i = 0; i < output.Dimensions[0]; i++)
            {
                var span = output.Buffer.Span.Slice(i * output.Strides[0]);
                var label = _model.Labels[(int)span[5]];
                var score = span[6];

                if (score < _model.Confidence) continue;  // Skip detections below confidence threshold

                var xMin = (span[1] - xPad) / gain;
                var yMin = (span[2] - yPad) / gain;
                var xMax = (span[3] - xPad) / gain;
                var yMax = (span[4] - yPad) / gain;

                var prediction = new YoloPrediction
                {
                    Label = label,
                    Score = score,
                    Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                };

                predictions.Add(prediction);
            }

            return predictions;
        }


        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(Image img)
        {
            Bitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height); // fit image size to specified input size
            }
            else
            {
                resized = img as Bitmap ?? new Bitmap(img);
            }

            var inputs = new[] // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized))
            };

            return _inferenceSession.Run(inputs, _model.Outputs); // run inference
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
