# Yolov7net

[![GitHub stars](https://img.shields.io/github/stars/ivilson/Yolov7net?style=social)](https://github.com/ivilson/Yolov7net/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ivilson/Yolov7net?style=social)](https://github.com/ivilson/Yolov7net/network)
[![GitHub issues](https://img.shields.io/github/issues/ivilson/Yolov7net)](https://github.com/ivilson/Yolov7net/issues)
[![GitHub license](https://img.shields.io/github/license/ivilson/Yolov7net)](https://github.com/ivilson/Yolov7net/blob/main/LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/ivilson/Yolov7net)](https://github.com/ivilson/Yolov7net/commits/main)
[![NuGet](https://img.shields.io/nuget/v/IVilson.AI.Yolov7net.svg)](https://www.nuget.org/packages/IVilson.AI.Yolov7net)
[![Platform](https://img.shields.io/badge/platform-.NET-blueviolet)](https://dotnet.microsoft.com/)

## Release Notes

### 2025/8/8
## 1.0.15 released

**Major Bug Fix: Improved Detection Accuracy** 🎯

Fixed a critical issue with bounding box coordinate accuracy that was affecting all YOLO model versions:

**🔧 What was fixed:**
- **Image preprocessing issue**: The original `ResizeImage` method was directly stretching images to target dimensions, causing image distortion and inaccurate detection results
- **Missing letterbox processing**: YOLO models require proper letterbox preprocessing to maintain aspect ratio

**✨ Improvements:**
- **New letterbox implementation**: Images are now resized while maintaining aspect ratio with proper padding
- **Consistent preprocessing**: All YOLO versions (v5, v7, v8, v9, v10, v11, v12) now use the same improved preprocessing
- **Better coordinate accuracy**: Bounding boxes now align correctly with detected objects
- **Standard YOLO preprocessing**: Uses gray padding (RGB: 128,128,128) as per YOLO standards

**🧪 Technical Details:**
```csharp
// New letterbox processing maintains aspect ratio
var scale = Math.Min(scaleWidth, scaleHeight);
var newWidth = (int)(sourceWidth * scale);
var newHeight = (int)(sourceHeight * scale);
var padX = (targetWidth - newWidth) / 2;
var padY = (targetHeight - newHeight) / 2;
```

**📊 Impact:**
- Significantly improved detection accuracy across all model versions
- More precise bounding box positioning
- Better consistency between different YOLO model predictions

This fix addresses the coordinate transformation issues that were causing detection boxes to appear in incorrect positions relative to the actual objects in the image.

### 2025/4/18
## 1.0.12 released

添加 yolov11 和 yolov12 的支持.重构项目结构，减少冗余存储。

## 2024.6.21
## 1.0.10 released

Fixed the bug where YOLOv10 inference results only contained one prediction.
修正了 yolov10 推理结果只有一个的bug.

![](https://raw.githubusercontent.com/ivilson/Yolov7net/master/demo_result_yolov10.jpg)


### 2024.6.12

修正了了使用 skiasharp 预测框不准的问题。

移除system.drawing 的支持

主分支已使用 skiasharp.

具体使用方法参考 demo 工程的 [Program.cs](https://github.com/ivilson/Yolov7net/blob/master/Yolov7net.Demo/Program.cs)

# 最新的性能测试 Performance 

work on i13900k + 64Gb Ram + RTX4090

![](https://raw.githubusercontent.com/ivilson/Yolov7net/master/performance.png)


![](https://raw.githubusercontent.com/ivilson/Yolov7net/master/test/Yolov7net.test/Assets/demo.jpg)

### 2024.6.9

Usage:

1. install-package IVilson.AI.Yolov7net
2. [Program.cs](https://github.com/ivilson/Yolov7net/blob/master/Yolov7net.Demo/Program.cs)
3. add yolov10 support.
Yolov10
```csharp
// init Yolov8 with onnx (include nms results)file path
using var yolo = new Yolov10("./assets/yolov10n.onnx", true);
// setup labels of onnx model 
yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
using var image = SKBitmap.Decode("Assets/" + fileName);
var predictions = yolo.Predict(image);  // now you can use numsharp to parse output data like this : var ret = yolo.Predict(image,useNumpy:true);
// draw box
using var canvas = new SKCanvas(image);
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

foreach (var prediction in predictions) // 迭代预测结果并绘制
{
    double score = Math.Round(prediction.Score, 2);
    paintRect.Color = prediction.Label.Color;
    paintText.Color = prediction.Label.Color;

    // 绘制矩形
    canvas.DrawRect(prediction.Rectangle, paintRect);

    // 绘制文本
    var x = prediction.Rectangle.X - 3;
    var y = prediction.Rectangle.Y - 23;
    canvas.DrawText($"{prediction.Label.Name} ({score})", x, y, paintText);
}
```


![](https://raw.githubusercontent.com/ivilson/Yolov7net/master/result.jpg)

yolov10 和 yolov7 保持兼容，包含了NMS 操作，感觉性能上比不上yolov9


### 2024.5.3 new branch released.

1. Considering cross-platform compatibility, such as supporting mobile development, I have removed numpy support in this new branch to reduce the size of the program package.
2. Remove System.Drawing and replace it with SkiaSharp.



------------------------------------------------------------
### HISTORY
# 2024.4.7 this project upgrade to net8.0

Repository Update Notice
As part of our ongoing efforts to improve our project structure and workflow, we have made the following changes to the branches of this repository:

The master branch has been renamed to net6.0. This change reflects our progression and the versioning aligned with the new features and improvements.
The net8.0 branch has been renamed to master. This is now the main branch where the latest stable releases and active developments will happen.



# Yolov7net Now support yolov9,yolov8,yolov7,yolov5.

.net 6 yolov5, yolov7, yolov8 onnx runtime interface, work for:
1. yolov9 https://github.com/WongKinYiu/yolov9
2. yolov8 https://github.com/ultralytics/ultralytics
3. yolov7 https://github.com/WongKinYiu/yolov7
4. yolov5 https://github.com/ultralytics/yolov5







# References & Acknowledgements

https://github.com/THU-MIG/yolov10

https://github.com/WongKinYiu/yolov9

https://github.com/ultralytics/ultralytics

https://github.com/WongKinYiu/yolov7

https://github.com/ultralytics/yolov5

https://github.com/mentalstack/yolov5-net

https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
