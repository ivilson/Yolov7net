using SkiaSharp;
using System.Drawing;

namespace Yolov7net
{
    public class YoloLabel
    {
        public int Id { get; set; }

        public string? Name { get; set; }
        
        public YoloLabelKind Kind { get; set; }
        
        public SKColor Color { get; set; }

        public YoloLabel() => Color = Color.Red;
    }
}
