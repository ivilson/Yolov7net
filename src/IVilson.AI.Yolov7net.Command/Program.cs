using SkiaSharp;
using System.CommandLine;
using System.Text.Json;


namespace Yolov7net
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            var useCudaOption = new Option<bool>(
                "--useCuda",
                getDefaultValue: () => false,
                "use cuda or not");

            var modelPathOption = new Option<FileInfo?>(
                "--modelPath",
                "Path to the YOLO model");

            var imagePathOption = new Option<FileInfo?>(
                "--imagePath",
                "Path to the input image");

            var confidenceThresholdOption = new Option<float>(
                "--confidenceThreshold",
                getDefaultValue: () => 0.5f,
                description: "Confidence threshold");

            var iouThresholdOption = new Option<float>(
                "--iouThreshold",
                getDefaultValue: () => 0.5f,
                description: "IoU threshold");

            var outputPathOption = new Option<FileInfo?>(
                "--outputPath",
                "Path to the output image");

            var customLabelsPathOption = new Option<FileInfo?>(
                "--labelsPath",
                "If your model is for custom labels, this should be your custom labels JSON file or use --labels. You can not use both.");

            var customLabelsOption = new Option<string>(
                "--labels",
                "If your model is for custom labels, set the labels like: --labels dog,cat. Or you can use --labelsPath. Choose one, you can not use both.");

            var yoloVersionOption = new Option<string>(
                "--yoloVersion",
                getDefaultValue: () => "v10",
                description: "The version of YOLO model to use. Default is v10.It support v5,v7,v8,v9,v10");


            var rootCommand = new RootCommand("YOLO.NET CLI for object detection")
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

            rootCommand.SetHandler((useCuda,modelPath, imagePath, confidenceThreshold, iouThreshold, outputPath, labelsPath, labels, yoloVersion) =>
            {
                if (modelPath == null || imagePath == null || outputPath == null)
                {
                    Console.WriteLine("Invalid arguments. Use --help for usage information.");
                    return;
                }

                if (labelsPath != null && !string.IsNullOrEmpty(labels))
                {
                    Console.WriteLine("You can use either --labelsPath or --labels, but not both.");
                    return;
                }

                PerformInference(useCuda, modelPath, imagePath, confidenceThreshold, iouThreshold, outputPath, labelsPath, labels, yoloVersion);
            },
            useCudaOption, modelPathOption, imagePathOption, confidenceThresholdOption, iouThresholdOption, outputPathOption, customLabelsPathOption, customLabelsOption , yoloVersionOption);

            return await rootCommand.InvokeAsync(args);
        }

        static void PerformInference(bool useCuda, FileInfo modelPath, FileInfo imagePath, float confidenceThreshold, float iouThreshold, FileInfo outputPath, FileInfo? labelsPath, string? labels,string yoloVersion)
        {
            IYoloNet yoloNet = null;
            switch(yoloVersion)
            {
                case "v5":
                    if(useCuda)
                    {
                        yoloNet = new Yolov5(modelPath.FullName, true);
                    }
                    else
                    {
                        yoloNet = new Yolov5(modelPath.FullName);
                    }
                    break;
                case "v7":
                    if (useCuda)
                    {
                        yoloNet = new Yolov7(modelPath.FullName, true);
                    }
                    else
                    {
                        yoloNet = new Yolov7(modelPath.FullName);
                    }
                    break;
                case "v8":
                    if (useCuda)
                    {
                        yoloNet = new Yolov8(modelPath.FullName, true);
                    }
                    else
                    {
                        yoloNet = new Yolov8(modelPath.FullName);
                    }
                    break;
                case "v9":
                    if (useCuda)
                    {
                        yoloNet = new Yolov9(modelPath.FullName, true);
                    }
                    else
                    {
                        yoloNet = new Yolov9(modelPath.FullName);
                    }
                    break;
                case "v10":
                    if (useCuda)
                    {
                        yoloNet = new Yolov10(modelPath.FullName, true);
                    }
                    else
                    {
                        yoloNet = new Yolov10(modelPath.FullName);
                    }
                    break;
                default:
                    Console.WriteLine("Invalid YOLO version. Supported versions are v5,v7,v8,v9,v10");
                    return;
            }


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
                // 加载自定义标签文件
                var customLabels = LoadLabelsFromFile(labelsPath.FullName);
                yoloNet.SetupLabels(customLabels);
            }
            else if (!string.IsNullOrEmpty(labels))
            {
                // 使用命令行中的自定义标签
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

            foreach (var prediction in predictions) // 迭代预测结果并绘制
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
            using var imageStream = new SKFileWStream(Path.Combine(outputPath.FullName, "result.json"));
            image.Encode(imageStream, SKEncodedImageFormat.Jpeg, 100);
            Console.WriteLine("Inference completed.");
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

        static void SaveResults(List<YoloPrediction> results, string outputPath)
        {
            try
            {
                var jsonString = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(Path.Combine(outputPath, "result.json"), jsonString);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving results: {ex.Message}");
            }
        }
    }
}


