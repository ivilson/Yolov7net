using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System.Collections.Concurrent;
using Yolov7net.Extentions;
using Yolov7net.Models;

namespace Yolov7net
{
    public class Yolov11 : Yolov8
    {


        public Yolov11(string modelPath, bool useCuda = false) : base(modelPath, useCuda)
        {
        }
    }
}
