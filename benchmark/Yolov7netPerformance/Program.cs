using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using SkiaSharp;
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
<<<<<<< HEAD
    private readonly Yolov9 _yolov9 = new Yolov9("./assets/yolov9-c.onnx", true);
    private readonly Yolov10 _yolov10 = new Yolov10("./assets/yolov9-c.onnx", true);
    private readonly Image _image = Image.FromFile("Assets/2dog.jpg");
    private readonly Image _image640;
=======
    private readonly Yolov8 _yolov9 = new Yolov8("./assets/yolov9-c.onnx", true);
    private readonly SKBitmap _image = SKBitmap.Decode("Assets/2dog.jpg");
    private readonly SKBitmap _image640;
>>>>>>> 5ab4336f34da713b03466ef167679565ea3d0046
    
    public YoloDetector()
    {
        _yolov7.SetupYoloDefaultLabels();
        _yolov8.SetupYoloDefaultLabels();
        _yolov5.SetupYoloDefaultLabels();
        _yolov9.SetupYoloDefaultLabels();
        _yolov10.SetupYoloDefaultLabels();
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
    public void Yolov9()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov8.Predict(_image);
        }
    }

    [Benchmark]
    public void Yolov9Resized()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov8.Predict(_image640);
        }
    }

<<<<<<< HEAD
    [Benchmark]
    public void Yolov10()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov10.Predict(_image);
        }
    }

    [Benchmark]
    public void Yolov10Resized()
    {
        for (int i = 0; i < RunCount; i++)
        {
            var ret = _yolov10.Predict(_image640);
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
=======
    //[Benchmark]
    //public void Yolov8Numpy()
    //{
    //    for (int i = 0; i < RunCount; i++)
    //    {
    //        var ret = _yolov8.Predict(_image, useNumpy: true);
    //    }
    //}
>>>>>>> 5ab4336f34da713b03466ef167679565ea3d0046
}

public class Program
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkRunner.Run(typeof(Program).Assembly);
        Console.ReadLine();
    }
}
