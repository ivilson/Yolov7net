// See https://aka.ms/new-console-template for more information
using System.Diagnostics;
using System.Drawing;
using Yolov7net;

using var yolo = new Yolov7("./assets/yolov7-tiny.onnx");
yolo.SetupYoloDefaultLabels();
Stopwatch sw = new Stopwatch();
sw.Start();
for (int i = 0; i < 10; i++)
{
    using var image = Image.FromFile("Assets/demo.jpg");
    var ret = yolo.Predict(image);
}
sw.Stop();
Console.WriteLine(sw.ElapsedMilliseconds);


using var yolov5 = new Yolov5("./assets/yolov7-tiny_640x640.onnx");
yolov5.SetupYoloDefaultLabels();
sw.Restart();
for (int i = 0; i < 10; i++)
{

    // setup labels of onnx model 
       // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
    using var image = Image.FromFile("Assets/demo.jpg");
    var ret = yolov5.Predict(image);
}
sw.Stop();

Console.WriteLine(sw.ElapsedMilliseconds);
Console.ReadLine();

