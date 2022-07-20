# Yolov7net

.net 6 yolov7 onnx runtime interface, work for https://github.com/WongKinYiu/yolov7

Usage:

![](https://github.com/iwaitu/Yolov7net/raw/master/Yolov7net.test/Assets/demo.jpg)


```csharp
// init Yolov7 with onnx file path
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
![](https://github.com/iwaitu/Yolov7net/raw/master/result.jpg)
# References & Acknowledgements

https://github.com/mentalstack/yolov5-net
