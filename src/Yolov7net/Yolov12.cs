using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Yolov7net;

namespace Yolov7net
{
    public class Yolov12 : Yolov8
    {
        public Yolov12(string modelPath, bool useCuda = false) : base(modelPath, useCuda)
        {
        }
    }
}
