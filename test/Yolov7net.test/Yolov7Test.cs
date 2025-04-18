using SkiaSharp;
using Yolov7net.Extentions;


namespace Yolov7net.test
{
    public class Yolov7Test
    {
        private readonly (SKBitmap image, string label)[] _testImages;

        public Yolov7Test()
        {
            var testFiles = new (string fileName, string label)[]
            {
                ("demo.jpg", "dog"),
                ("cat_224x224.jpg", "cat"),
            };

            var array = new (SKBitmap image, string label)[testFiles.Length * 2];
            int i = 0;
            foreach (var tuple in testFiles)
            {
                var image = SKBitmap.Decode("Assets/" + tuple.fileName);
                array[i++] = (image, tuple.label);

                // resized image should give the same result
                image = Utils.ResizeImage(image, 640, 640);
                array[i++] = (image, tuple.label);
            }

            _testImages = array;
        }

        [Fact]
        public void SKBitmapTest()
        {
            using var yolo = new Yolov8("./assets/yolov8n.onnx");
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);

            var ret = yolo.Predict(_testImages[0].image);
            CheckResult(ret, _testImages[0].label);
        }

        [Fact]
        public void TestYolov7()
        {
            using var yolo = new Yolov7("./assets/yolov7-tiny.onnx", true); 
            
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
            using var yolo = new Yolov5("./assets/yolov7-tiny_640x640.onnx"); 
            
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
            using var yolo = new Yolov8("./assets/yolov8n.onnx",useCuda:true); 
            
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
        public void TestYolov9()
        {
            using var yolo = new Yolov9("./assets/yolov9-c.onnx"); 

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
        public void TestYolov10()
        {
            using var yolo = new Yolov10("./assets/yolov10n.onnx"); 

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
        public void TestYolov11()
        {
            using var yolo = new Yolov11("./assets/yolo11n.onnx");

            // setup labels of onnx model 
            yolo.SetupYoloDefaultLabels();   // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            Assert.NotNull(yolo);

            foreach (var tuple in _testImages)
            {
                var ret = yolo.Predict(tuple.image);
                CheckResult(ret, tuple.label);
            }
        }

        private void CheckResult(List<YoloPrediction> predictions, string label)
        {
            Assert.NotNull(predictions);
            Assert.Equal(1, predictions.Where(p=>p.Score > 0.5).ToList() .Count);
            Assert.Equal(label, predictions.Where(p => p.Score > 0.5).FirstOrDefault().Label.Name);
            //Debug.WriteLine(predictions[0].Rectangle);
        }
    }
}
