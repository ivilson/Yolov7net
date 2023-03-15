// See https://aka.ms/new-console-template for more information
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Drawing;
using Yolov7net;

[MemoryDiagnoser]
public class YoloDetector
{
    private Yolov7 yolov7 = new Yolov7("./assets/yolov7-tiny.onnx",true);
    private Yolov5 yolov5 = new Yolov5("./assets/yolov7-tiny_640x640.onnx",true);
    private Yolov8 yolov8 = new Yolov8("./assets/yolov8n.onnx", true);
    private Image image = Image.FromFile("Assets/2dog.jpg");
    
    public YoloDetector()
    {

        yolov7.SetupYoloDefaultLabels();
        yolov8.SetupYoloDefaultLabels();
        yolov5.SetupYoloDefaultLabels();
    }

    [Benchmark]
    public void Yolov7()
    {
        var ret = yolov7.Predict(image);
    }

    [Benchmark]
    public void Yolov8()
    {
        var ret = yolov8.Predict(image);
    }

    [Benchmark]
    public void Yolov8Numpy()
    {
        var ret = yolov8.Predict(image, useNumpy: true);
    }

    [Benchmark]
    public void Yolov5()
    {
        var ret = yolov5.Predict(image);
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkRunner.Run(typeof(Program).Assembly);
        Console.ReadLine();
    }
}
