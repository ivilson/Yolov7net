### 2024.5.3 new branch released.

1. Considering cross-platform compatibility, such as supporting mobile development, I have removed numpy support in this new branch to reduce the size of the program package.
2. Remove System.Drawing and replace it with SkiaSharp.



------------------------------------------------------------
### HISTORY


2024.4.7 this project upgrade to net8.0

Repository Update Notice
As part of our ongoing efforts to improve our project structure and workflow, we have made the following changes to the branches of this repository:

The master branch has been renamed to net6.0. This change reflects our progression and the versioning aligned with the new features and improvements.
The net8.0 branch has been renamed to master. This is now the main branch where the latest stable releases and active developments will happen.


# Performance 

work on i13900k + 64Gb Ram + RTX4090

![](https://github.com/iwaitu/Yolov7net/raw/master/performance.png)



# Yolov7net Now support yolov9,yolov8,yolov7,yolov5.

.net 6 yolov5, yolov7, yolov8 onnx runtime interface, work for:
1. yolov9 https://github.com/WongKinYiu/yolov9
2. yolov8 https://github.com/ultralytics/ultralytics
3. yolov7 https://github.com/WongKinYiu/yolov7
4. yolov5 https://github.com/ultralytics/yolov5


Usage:

install-package IVilson.AI.Yolov7net

![](https://github.com/ivilson/Yolov7net/raw/master/test/Yolov7net.test/Assets/demo.jpg)

yolov9 和 yolov8 的 onnx 输出参数 (1,84,8400) ，调整了输出结果的 ndarray 的结构

如果有问题请前往 issus 进行提问，我会尽量解答

Yolov9
```csharp
// init Yolov8 with onnx (include nms results)file path
using var yolo = new Yolov8("./assets/yolov9-c.onnx", true);
// setup labels of onnx model 
yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
using var image = Image.FromFile("Assets/demo.jpg");
var predictions = yolo.Predict(image);  // now you can use numsharp to parse output data like this : var ret = yolo.Predict(image,useNumpy:true);
// draw box
using var graphics = Graphics.FromImage(image);
foreach (var prediction in predictions) // iterate predictions to draw results
{
    double score = Math.Round(prediction.Score, 2);
    graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),new[] { prediction.Rectangle });
    var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);
    graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                    new PointF(x, y));
}
```

Yolov8
```csharp
// init Yolov8 with onnx (include nms results)file path
using var yolo = new Yolov8("./assets/yolov8n.onnx", true);
// setup labels of onnx model 
yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
using var image = Image.FromFile("Assets/demo.jpg");
var predictions = yolo.Predict(image);  // now you can use numsharp to parse output data like this : var ret = yolo.Predict(image,useNumpy:true);
// draw box
using var graphics = Graphics.FromImage(image);
foreach (var prediction in predictions) // iterate predictions to draw results
{
    double score = Math.Round(prediction.Score, 2);
    graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),new[] { prediction.Rectangle });
    var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);
    graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                    new PointF(x, y));
}
```
yolov7 可以直接导出包含nms操作结果的onnx， 使用方法略有不同，需要使用 Yolov7 这个类

```csharp
// init Yolov7 with onnx (include nms results)file path
using var yolo = new Yolov7("./assets/yolov7-tiny_640x640.onnx", true);
// setup labels of onnx model 
yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
using var image = Image.FromFile("Assets/demo.jpg");
var predictions = yolo.Predict(image);

// draw box
using var graphics = Graphics.FromImage(image);
foreach (var prediction in predictions) // iterate predictions to draw results
{
    double score = Math.Round(prediction.Score, 2);
    graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),new[] { prediction.Rectangle });
    var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);
    graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                    new PointF(x, y));
}
```

对于未包括nms 结果的模型，需要用到 yolov5 这个类
```csharp
// init Yolov5 with onnx file path
using var yolo = new Yolov5("./assets/yolov7-tiny_640x640.onnx", true);
// setup labels of onnx model 
yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
using var image = Image.FromFile("Assets/demo.jpg");
var predictions = yolo.Predict(image);

// draw box
using var graphics = Graphics.FromImage(image);
foreach (var prediction in predictions) // iterate predictions to draw results
{
    double score = Math.Round(prediction.Score, 2);
    graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),new[] { prediction.Rectangle });
    var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);
    graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                    new PointF(x, y));
}


```
![](https://github.com/iwaitu/Yolov7net/raw/master/result.jpg)
# References & Acknowledgements

https://github.com/WongKinYiu/yolov9

https://github.com/ultralytics/ultralytics

https://github.com/WongKinYiu/yolov7

https://github.com/ultralytics/yolov5

https://github.com/mentalstack/yolov5-net

https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection

