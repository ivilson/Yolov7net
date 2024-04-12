# Performance 

work on i13900k + 64Gb Ram + RTX4090

![](https://github.com/iwaitu/Yolov7net/raw/master/performance.png)



# Yolov7net Now support yolov5,yolov7,yolov8,yolov9.

.net 6 yolov5, yolov7, yolov8 onnx runtime interface, work for:
1. yolov9 https://github.com/WongKinYiu/yolov9
2. yolov8 https://github.com/ultralytics/ultralytics
3. yolov7 https://github.com/WongKinYiu/yolov7
4. yolov5 https://github.com/ultralytics/yolov5


Usage:

install-package IVilson.AI.Yolov7net

![](https://github.com/ivilson/Yolov7net/raw/master/test/Yolov7net.test/Assets/demo.jpg)

支持的yolo 版本包括 yolov5,yolov7,yolov8,yolov9 

如果有问题请前往 issus 进行提问，我会尽量解答

yolov9 和 yolov8 保持兼容，所以当你使用 yolov9 的 onnx 模型时，可以继续使用 new Yolov8("xxx.onnx") 的方式加载

```csharp
// init Yolov8 with onnx (include nms results)file path
using var yolo = new Yolov8("./assets/yolov7-tiny_640x640.onnx", true);
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

https://github.com/ultralytics/ultralytics

https://github.com/WongKinYiu/yolov7

https://github.com/ultralytics/yolov5

https://github.com/mentalstack/yolov5-net

https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection

