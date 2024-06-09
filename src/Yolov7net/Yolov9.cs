namespace Yolov7net
{
    public class Yolov9 : Yolov8
    {
        public Yolov9(string modelPath, bool useCuda = false) : base(modelPath, useCuda)
        {
        }
    }
}
