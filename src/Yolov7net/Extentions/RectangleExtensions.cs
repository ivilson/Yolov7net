using System.Drawing;

namespace Yolov7net.Extentions
{
    public static class RectangleExtensions
    {
        public static float Area(this RectangleF source)
        {
            return source.Width * source.Height;
        }
    }
}
