using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Yolov7net.Models
{
    public class YoloModel
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public int Depth { get; set; }

        public int Dimensions { get; set; } //yolov7 包含nms 的模型不需要此参数
        public float Confidence { get; set; } = 0.20f; //yolov7 包含nms 的模型不需要此参数
        public float MulConfidence { get; set; } = 0.25f; //yolov7 包含nms 的模型不需要此参数
        public float Overlap { get; set; } = 0.45f; //yolov7 包含nms 的模型不需要此参数
        public string[] Outputs { get; set; }
        public List<YoloLabel> Labels { get; set; } = new List<YoloLabel>();
        public bool UseDetect { get; set; }
    }
}
