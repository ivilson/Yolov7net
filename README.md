### 2024.6.21
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

