using IVilson.AI.Yolov7net.Command;
using SkiaSharp;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Yolov7net
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            var useCudaOption = new Option<bool>(
                name:"--useCuda",
                getDefaultValue: () => true,
                description: "Use CUDA or not");

            var modelPathOption = new Option<FileInfo?>(
                name: "--modelPath",
                description: "Path to the YOLO model");

            var imagePathOption = new Option<FileInfo?>(
                name: "--imagePath",
                description: "Path to the input image");

            var confidenceThresholdOption = new Option<float>(
                name: "--confidenceThreshold",
                getDefaultValue: () => 0.5f,
                description: "Confidence threshold");

            var iouThresholdOption = new Option<float>(
                name: "--iouThreshold",
                getDefaultValue: () => 0.5f,
                description: "IoU threshold");

            var outputPathOption = new Option<DirectoryInfo?>(
                name: "--outputPath",
                "Path to the output directory");

            var customLabelsPathOption = new Option<FileInfo?>(
                name: "--labelsPath",
                "If your model is for custom labels, this should be your custom labels JSON file or use --labels. You can not use both.");

            var customLabelsOption = new Option<string>(
                name: "--labels",
                "If your model is for custom labels, set the labels like: --labels dog,cat. Or you can use --labelsPath. Choose one, you can not use both.");

            var yoloVersionOption = new Option<string>(
                name: "--yoloVersion",
                getDefaultValue: () => "v10",
                description: "The version of YOLO model to use. Default is v10. It supports v5, v7, v8, v9, v10");

            var rootCommand = new RootCommand("YOLO.NET CLI for object detection");

            var predictCommand = new Command(name: "predict", "predict image")
                {
                    useCudaOption,
                    modelPathOption,
                    imagePathOption,
                    confidenceThresholdOption,
                    iouThresholdOption,
                    outputPathOption,
                    customLabelsPathOption,
                    customLabelsOption,
                    yoloVersionOption
                };
            rootCommand.AddCommand(predictCommand);

            predictCommand.SetHandler(async (InvocationContext context) =>
            {
                var useCuda = context.ParseResult.GetValueForOption(useCudaOption);
                var modelPath = context.ParseResult.GetValueForOption(modelPathOption);
                var imagePath = context.ParseResult.GetValueForOption(imagePathOption);
                var confidenceThreshold = context.ParseResult.GetValueForOption(confidenceThresholdOption);
                var iouThreshold = context.ParseResult.GetValueForOption(iouThresholdOption);
                var outputPath = context.ParseResult.GetValueForOption(outputPathOption);
                var customLabelsPath = context.ParseResult.GetValueForOption(customLabelsPathOption);
                var customLabels = context.ParseResult.GetValueForOption(customLabelsOption);
                var yoloVersion = context.ParseResult.GetValueForOption(yoloVersionOption);

                // 在这里处理你的参数
                if(modelPath == null || imagePath == null || outputPath == null)
                {
                    Console.WriteLine("Model path, image path and output path are required.");
                    return;
                }
                Console.WriteLine($"Use CUDA: {useCuda}");
                Console.WriteLine($"Model Path: {modelPath?.FullName}");
                Console.WriteLine($"Image Path: {imagePath?.FullName}");
                Console.WriteLine($"Confidence Threshold: {confidenceThreshold}");
                Console.WriteLine($"IoU Threshold: {iouThreshold}");
                Console.WriteLine($"Output Path: {outputPath?.FullName}");
                Console.WriteLine($"Labels Path: {customLabelsPath?.FullName}");
                Console.WriteLine($"Labels: {customLabels}");
                Console.WriteLine($"YOLO Version: {yoloVersion}");
                await Task.Run(() => {
                    PerformInference(useCuda, modelPath, imagePath, confidenceThreshold, iouThreshold, outputPath, customLabelsPath, customLabels, yoloVersion);
                });
            });

            return await rootCommand.InvokeAsync(args);
        }

        static int Test(bool? useCuda)
        {
            return 0;
        }

        static void PerformInference(bool useCuda, FileInfo modelPath, FileInfo imagePath, float confidenceThreshold, float iouThreshold, DirectoryInfo outputPath, FileInfo? labelsPath, string? labels, string yoloVersion)
        {
            try
            {
                IYoloNet yoloNet = yoloVersion switch
                {
                    "v5" => useCuda ? new Yolov5(modelPath.FullName, true) : new Yolov5(modelPath.FullName),
                    "v7" => useCuda ? new Yolov7(modelPath.FullName, true) : new Yolov7(modelPath.FullName),
                    "v8" => useCuda ? new Yolov8(modelPath.FullName, true) : new Yolov8(modelPath.FullName),
                    "v9" => useCuda ? new Yolov9(modelPath.FullName, true) : new Yolov9(modelPath.FullName),
                    "v10" => useCuda ? new Yolov10(modelPath.FullName, true) : new Yolov10(modelPath.FullName),
                    _ => throw new ArgumentException("Invalid YOLO version. Supported versions are v5, v7, v8, v9, v10")
                };

                // 加载图像
                var image = SKBitmap.Decode(imagePath.FullName);
                var canvas = new SKCanvas(image);
                var paintRect = new SKPaint
                {
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = 1,
                    IsAntialias = true
                };

                var paintText = new SKPaint
                {
                    TextSize = 16,
                    IsAntialias = true,
                    Typeface = SKTypeface.FromFamilyName("Consolas")
                };

                // 加载标签
                if (labelsPath != null)
                {
                    var customLabels = LoadLabelsFromFile(labelsPath.FullName);
                    yoloNet.SetupLabels(customLabels);
                }
                else if (!string.IsNullOrEmpty(labels))
                {
                    var customLabels = labels.Split(',');
                    yoloNet.SetupLabels(customLabels);
                }
                else
                {
                    yoloNet.SetupYoloDefaultLabels();
                }

                // 进行推理
                var predictions = yoloNet.Predict(image);

                // 处理结果并保存输出
                SaveResults(predictions, outputPath.FullName);

                foreach (var prediction in predictions)
                {
                    double score = Math.Round(prediction.Score, 2);

                    // 绘制矩形
                    canvas.DrawRect(prediction.Rectangle, paintRect);

                    // 绘制文本
                    var x = prediction.Rectangle.Left + 3;
                    var y = prediction.Rectangle.Top + 23;
                    canvas.DrawText($"{prediction.Label.Name} ({score})", x, y, paintText);
                }
                canvas.Flush();

                //保存绘制结果
                using var imageStream = new SKFileWStream(Path.Combine(outputPath.FullName, "result.jpg"));
                image.Encode(imageStream, SKEncodedImageFormat.Jpeg, 100);
                Console.WriteLine("Inference completed.");
            }
            catch (Exception ex)
            {

                Console.WriteLine(ex.Message);
            }
            
        }

        static string[] LoadLabelsFromFile(string filePath)
        {
            try
            {
                var jsonString = File.ReadAllText(filePath);
                var jsonDocument = JsonDocument.Parse(jsonString);
                if (jsonDocument.RootElement.TryGetProperty("labels", out JsonElement labelsElement) && labelsElement.ValueKind == JsonValueKind.Array)
                {
                    var labelsList = new List<string>();
                    foreach (var label in labelsElement.EnumerateArray())
                    {
                        labelsList.Add(label.GetString() ?? string.Empty);
                    }
                    return labelsList.ToArray();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reading labels file: {ex.Message}");
            }
            return Array.Empty<string>();
        }

        static void SaveResults(IList<YoloPrediction> results, string outputPath)
        {
            try
            {
                var dtos = results.Select(r => new YoloPredictionDto
                {
                    Label = r.Label == null ? null : new YoloLabelDto
                    {
                        Id = r.Label.Id,
                        Name = r.Label.Name,
                        Color = r.Label.Color.ToString()
                    },
                    Rectangle = new SimpleRect
                    {
                        Left = r.Rectangle.Left,
                        Top = r.Rectangle.Top,
                        Right = r.Rectangle.Right,
                        Bottom = r.Rectangle.Bottom
                    },
                    Score = r.Score
                }).ToList();

                var options = new JsonSerializerOptions
                {
                    WriteIndented = true,
                    ReferenceHandler = ReferenceHandler.Preserve
                };

                var jsonString = JsonSerializer.Serialize(dtos, options);
                File.WriteAllText(Path.Combine(outputPath, "result.json"), jsonString);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving results: {ex.Message}");
            }
        }
    }

    public class CommandHandler : ICommandHandler
    {
        public int Invoke(InvocationContext context)
        {
            return 0;
        }

        public Task<int> InvokeAsync(InvocationContext context)
        {
            return Task.FromResult(Invoke(context));
        }
    }

}
