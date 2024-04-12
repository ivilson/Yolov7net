using System.Diagnostics;
using System.Drawing;
using Yolov7net.Extentions;


namespace Yolov7net.test
{
    public class Yolov7Test
    {
        private readonly (Image image, string label)[] _testImages;

        public Yolov7Test()
        {
            var testFiles = new (string fileName, string label)[]
            {
                ("demo.jpg", "dog"),
                ("cat_224x224.jpg", "cat"),
            };

            var array = new (Image image, string label)[testFiles.Length * 2];
            int i = 0;
            foreach (var tuple in testFiles)
            {
                var image = Image.FromFile("Assets/" + tuple.fileName);
                array[i++] = (image, tuple.label);

                // resized image should give the same result
                image = Utils.ResizeImage(image, 640, 640);
                array[i++] = (image, tuple.label);
            }

            _testImages = array;
        }

        [Fact]
        public void TestYolov7()
        {
            using var yolo = new Yolov7("./assets/yolov7-tiny.onnx", true); //yolov7 模型,不需要 nms 操作
            
            // setup labels of onnx model 
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);

            foreach (var tuple in _testImages)
            {
                var ret = yolo.Predict(tuple.image);
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
                var ret = yolo.Predict(tuple.image);
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
                var ret = yolo.Predict(tuple.image, useNumpy: false);
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
                var ret = yolo.Predict(tuple.image, useNumpy: true);
                CheckResult(ret, tuple.label);
            }
        }

        [Fact]
        public void TestYolov9()
        {
            using var yolo = new Yolov8("./assets/yolov9-c.onnx"); //yolov9 模型,需要 nms 操作

            // setup labels of onnx model 
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);

            foreach (var tuple in _testImages)
            {
                var ret = yolo.Predict(tuple.image, useNumpy: false);
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

        
    }
}
