// See https://aka.ms/new-console-template for more information
using SkiaSharp;
using System.Runtime.CompilerServices;
using Yolov7net;
using Yolov7net.Models;
using static System.Net.Mime.MediaTypeNames;

Console.WriteLine("Hello, World!");







var paintRect = new SKPaint
{
    Style = SKPaintStyle.Stroke,
    StrokeWidth = 2,
    IsAntialias = true,
    Color = SKColors.Green
};

var paintText = new SKPaint
{
    TextSize = 16,
    IsAntialias = true,
    Color = SKColors.Green,
    IsStroke = false

};


#region yolov5

using (var yolo = new Yolov5("./assets/yolov5-tiny_640x640.onnx", true)) 
    RunYolo(yolo, "yolov5");
}

#endregion


#region yolov7

using (var yolo = new Yolov7("./assets/yolov7-tiny.onnx", true))
{
    RunYolo(yolo, "yolov7");
}

#endregion

#region yolov8
using (var yolo = new Yolov8("./assets/yolov8n.onnx", true))
{
    RunYolo(yolo, "yolov8");
}

#endregion

#region yolov9
using (var yolo = new Yolov9("./assets/yolov9-c.onnx", true))
{
    RunYolo(yolo, "yolov9");
}

#endregion

#region yolov10
using (var yolo = new Yolov10("./assets/yolov10n.onnx", true))
{
   RunYolo(yolo, "yolov10");
}
   
#endregion

void RunYolo(IYoloNet yolo,string remark="")
{
    yolo.SetupYoloDefaultLabels();
    using var image = SKBitmap.Decode("Assets/demo.jpg");
    var predictions = yolo.Predict(image);

    using var canvas = new SKCanvas(image);
    foreach (var prediction in predictions) // 迭代预测结果并绘制
    {
        double score = Math.Round(prediction.Score, 2);


        // 绘制矩形
        canvas.DrawRect(prediction.Rectangle, paintRect);

        // 绘制文本
        var x = prediction.Rectangle.Left + 3;
        var y = prediction.Rectangle.Top + 23;
        canvas.DrawText($"{prediction.Label.Name} ({score})  {remark}", x, y, paintText);
    }


    canvas.Flush();

    //保存绘制结果
    using var imageStream = new SKFileWStream($"demo_result_{remark}.jpg");
    image.Encode(imageStream, SKEncodedImageFormat.Jpeg, 100);
    Console.WriteLine($"Done {remark}!");
}