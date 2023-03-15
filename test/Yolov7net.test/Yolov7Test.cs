using System;
using System.Diagnostics;
using System.Drawing;


namespace Yolov7net.test
{
    public class Yolov7Test
    {
        private readonly (string imageName, string label)[] _testImages = new (string imageName, string label)[]
        {
            ("demo.jpg", "dog"),
            ("cat_224x224.jpg", "cat"),
        };
        
        [Fact]
        public void TestYolov7()
        {
            using var yolo = new Yolov7("./assets/yolov7-tiny.onnx", true); //yolov7 模型,不需要 nms 操作
            
            // setup labels of onnx model 
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);

            foreach (var tuple in _testImages)
            {
                using var image = Image.FromFile("Assets/" + tuple.imageName);
                var ret = yolo.Predict(image);
                CheckResult(ret, tuple.label);
            }
        }

        [Fact]
        public void TestYolov5()
        {
            using var yolo = new Yolov5("./assets/yolov7-tiny_640x640.onnx"); //yolov5 or yolov7 模型, 需要 nms 操作
            
            // setup labels of onnx model 
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);

            foreach (var tuple in _testImages)
            {
                using var image = Image.FromFile("Assets/" + tuple.imageName);
                var ret = yolo.Predict(image);
                CheckResult(ret, tuple.label);
            }
        }

        [Fact]
        public void TestYolov8()
        {
            using var yolo = new Yolov8("./assets/yolov8n.onnx"); //yolov8 模型,需要 nms 操作
            
            // setup labels of onnx model 
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);

            foreach (var tuple in _testImages)
            {
                using var image = Image.FromFile("Assets/" + tuple.imageName);
                var ret = yolo.Predict(image, useNumpy: false);
                CheckResult(ret, tuple.label);
            }
        }

        [Fact]
        public void TestYolov8Numpy()
        {
            using var yolo = new Yolov8("./assets/yolov8n.onnx"); //yolov8 模型,需要 nms 操作

            // setup labels of onnx model 
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);

            foreach (var tuple in _testImages)
            {
                using var image = Image.FromFile("Assets/" + tuple.imageName);
                var ret = yolo.Predict(image, useNumpy: true);
                CheckResult(ret, tuple.label);
            }
        }

        private void CheckResult(List<YoloPrediction> predictions, string label)
        {
            Assert.NotNull(predictions);
            Assert.Equal(1, predictions.Count);
            Assert.Equal(label, predictions[0].Label.Name);
            System.Diagnostics.Debug.WriteLine(predictions[0].Rectangle);
        }

        [Fact]
        public void TestPerformance()
        {
            using var yolo = new Yolov7("./assets/yolov7-tiny.onnx");
            yolo.SetupYoloDefaultLabels();
            
            var sw = Stopwatch.StartNew();
            for(int i = 0; i < 10; i++)
            {
                using var image = Image.FromFile("Assets/demo.jpg");
                var ret = yolo.Predict(image);
            }
        
            sw.Stop();
            Debug.WriteLine(sw.ElapsedMilliseconds);
        }
    }
}
