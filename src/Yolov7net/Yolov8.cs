using System.Collections.Concurrent;
using System.Drawing;
using Yolov7net.Extentions;
using Yolov7net.Models;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using NumSharp;

namespace Yolov7net
{
    public class Yolov8 : IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private readonly YoloModel _model = new YoloModel();
        private bool _useNumpy;

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

        public List<YoloPrediction> Predict(Image image, float conf_thres = 0, float iou_thres = 0,bool useNumpy = false)
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

            _useNumpy = useNumpy;
            using var outputs = Inference(image);
            return Suppress(ParseOutput(outputs, image));
        }

        /// <summary>
        /// Removes overlapped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Suppress(List<YoloPrediction> items)
        {
            var areas = items.ToDictionary(item => item, item => item.Rectangle.Area());
            var toRemove = new ConcurrentBag<YoloPrediction>();

            Parallel.ForEach(items, item =>
            {
                foreach (var current in items)
                {
                    if (current == item || toRemove.Contains(current)) continue;

                    var intersection = RectangleF.Intersect(item.Rectangle, current.Rectangle);
                    if (intersection.IsEmpty) continue;

                    float intArea = intersection.Area();
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


        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Inference(Image img)
        {
            Bitmap resized;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height);
            }
            else
            {
                resized = img as Bitmap ?? new Bitmap(img);
            }

            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels2(resized))
            };

            return _inferenceSession.Run(inputs, _model.Outputs);
        }

        private List<YoloPrediction> ParseOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, Image image)
        {
            string firstOutput = _model.Outputs[0];
            var output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;

            if (_useNumpy)
            {
                return ParseDetectNumpy(output, image);
            }

            return ParseDetect(output, image);
        }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
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
                            Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
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




        private List<YoloPrediction> ParseDetectNumpy(DenseTensor<float> output, Image image)
        {
            float[] outputArray = output.ToArray();
            var numpyArray = np.array(outputArray, np.float32);
            var data = numpyArray.reshape(84, 8400).transpose(new int[] { 1, 0 });
            return ProcessResult(data, image);
        }

        private List<YoloPrediction> ProcessResult(NDArray data, Image image)
        {
            var result = new ConcurrentBag<YoloPrediction>();
            var scores = np.max(data[":, 4:"], axis: 1);

            var temp = data[scores > 0.2f];
            scores = scores[scores > 0.2f];
            var class_ids = np.argmax(temp[":, 4:"], 1);
            var boxes = extract_rect(temp, image.Width, image.Height);
            var indices = nms(boxes, scores);
            foreach (var x in indices)
            {
                var label = _model.Labels[class_ids[x]];
                var prediction = new YoloPrediction(label, scores[x])
                {
                    Rectangle = new RectangleF(boxes[x][0], boxes[x][1], boxes[x][2] - boxes[x][0], boxes[x][3] - boxes[x][1])
                };
                result.Add(prediction);
            };
            return result.ToList();
        }


        private int[] nms(NDArray boxes, NDArray scores, float iou_threshold = .5f)
        {

            // Sort by score
            var sortedIndices = np.argsort<float>(scores)["::-1"];

            List<int> keepBoxes = new List<int>();
            int[] sortedIndicesArray = sortedIndices.Data<int>().ToArray();
            while (sortedIndicesArray.Length > 0)
            {
                // Pick the last box
                int boxId = sortedIndicesArray[0];
                keepBoxes.Add(boxId);
                // Compute IoU of the picked box with the rest
                NDArray ious = ComputeIOU(boxes[boxId], boxes[sortedIndices["1:"]]);

                // Remove boxes with IoU over the threshold
                var keepIndices = ious.Data<float>().AsQueryable().ToArray().Select(x => x < iou_threshold).ToArray();
                sortedIndicesArray = sortedIndicesArray.Skip(keepIndices.Length + 1).ToArray();
            }

            return keepBoxes.ToArray();
        }

        private NDArray ComputeIOU(NDArray box, NDArray boxes)
        {
            // Compute xmin, ymin, xmax, ymax for both boxes
            var xmin = np.maximum(box[0], boxes[":", 0]);
            var ymin = np.maximum(box[1], boxes[":", 1]);
            var xmax = np.minimum(box[2], boxes[":", 2]);
            var ymax = np.minimum(box[3], boxes[":", 3]);

            // Compute intersection area
            var intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin);

            // Compute union area
            var box_area = (box[2] - box[0]) * (box[3] - box[1]);
            var boxes_area = (boxes[":", 2] - boxes[":", 0]) * (boxes[":", 3] - boxes[":", 1]);
            var union_area = box_area + boxes_area - intersection_area;

            // Compute IoU
            var iou = intersection_area / union_area;

            return iou;
        }

        private NDArray extract_rect(NDArray temp, int width, int height)
        {
            var data = rescale_boxes(temp[":, :4"], width, height);
            var boxes = Xywh2Xyxy(data);
            return boxes;
        }

        public NDArray Xywh2Xyxy(NDArray x)
        {
            var y = x.Clone();
            y[":", 0] = x[":", 0] - x[":", 2] / 2;
            y[":", 1] = x[":", 1] - x[":", 3] / 2;
            y[":", 2] = x[":", 0] + x[":", 2] / 2;
            y[":", 3] = x[":", 1] + x[":", 3] / 2;
            return y;
        }

        private NDArray rescale_boxes(NDArray boxes, int width, int height)
        {

            NDArray inputShape = np.array(new float[] { _model.Width, _model.Height, _model.Width, _model.Height });
            NDArray resizedBoxes = np.divide(boxes, inputShape);
            resizedBoxes = np.multiply(resizedBoxes, new float[] { width, height, width, height });
            return resizedBoxes;
        }

        private void prepare_input(Image img)
        {
            Bitmap bmp = Utils.ResizeImage(img, _model.Width, _model.Height);

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
