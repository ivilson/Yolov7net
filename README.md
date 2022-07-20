# Yolov7net

.net 6 yolov7 interface, work for https://github.com/WongKinYiu/yolov7

Usage

```csharp
// init Yolov7 with onnx file path
using var yolo = new Yolov7("./assets/yolov7-tiny_640x640.onnx", true);
// setup labels of onnx model 
yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
using var image = Image.FromFile("Assets/demo.jpg");
var ret = yolo.Predict(image);
```

# References:

https://github.com/mentalstack/yolov5-net
