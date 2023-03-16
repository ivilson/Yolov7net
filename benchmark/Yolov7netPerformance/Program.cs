using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Drawing;
using Yolov7net;
using Yolov7net.Extentions;

[MemoryDiagnoser]
public class YoloDetector
{
    private const int RunCount = 1;
    private readonly Yolov7 _yolov7 = new Yolov7("./assets/yolov7-tiny.onnx",true);
    private readonly Yolov5 _yolov5 = new Yolov5("./assets/yolov7-tiny_640x640.onnx",true);
    private readonly Yolov8 _yolov8 = new Yolov8("./assets/yolov8n.onnx", true);
    private readonly Image _image = Image.FromFile("Assets/2dog.jpg");
    private readonly Image _image640;
    
    public YoloDetector()
    {
        _yolov7.SetupYoloDefaultLabels();
        _yolov8.SetupYoloDefaultLabels();
        _yolov5.SetupYoloDefaultLabels();
        _image640 = Utils.ResizeImage(_image, 640, 640);
    }

    [Benchmark]
    public void Yolov5()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov5.Predict(_image);
        }
    }

    [Benchmark]
    public void Yolov5Resized()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov5.Predict(_image640);
        }
    }

    [Benchmark]
    public void Yolov7()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov7.Predict(_image);
        }
    }

    [Benchmark]
    public void Yolov7Resized()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov7.Predict(_image640);
        }
    }

    [Benchmark]
    public void Yolov8()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov8.Predict(_image);
        }
    }

    [Benchmark]
    public void Yolov8Resized()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov8.Predict(_image640);
        }
    }

    [Benchmark]
    public void Yolov8Numpy()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov8.Predict(_image, useNumpy: true);
        }
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        new YoloDetector().Yolov7Resized();
        var summary = BenchmarkRunner.Run(typeof(Program).Assembly);
        Console.ReadLine();
    }
}
