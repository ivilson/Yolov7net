
using System.Drawing;

namespace Yolov7net.test
{
    public class Yolov7Test
    {
        [Fact]
        public void InitTest()
        {
            // init Yolov7 with onnx file path
            using var yolo = new Yolov7("./assets/yolov7-tiny_640x640.onnx", true);
            // setup labels of onnx model 
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);
            using var image = Image.FromFile("Assets/demo.jpg");
            var ret = yolo.Predict(image);
            Assert.NotNull(ret);
            Assert.True(ret.Count == 1);
        }
    }
}