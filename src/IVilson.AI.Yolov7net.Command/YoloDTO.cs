namespace IVilson.AI.Yolov7net.Command
{
    public class YoloPredictionDto
    {
        public YoloLabelDto? Label { get; set; }
        public SimpleRect Rectangle { get; set; }
        public float Score { get; set; }
    }

    public class YoloLabelDto
    {
        public int Id { get; set; }
        public string? Name { get; set; }
        public string Color { get; set; }
    }

    public class SimpleRect
    {
        public float Left { get; set; }
        public float Top { get; set; }
        public float Right { get; set; }
        public float Bottom { get; set; }
    }

}
