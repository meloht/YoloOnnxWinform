using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloOnnxWinform.YoloOnnx;

namespace YoloOnnxWinform.YoloWarpper
{
    public class YoloDetectImpl : YoloDetectImplBase,IYoloModel
    {
        private YoloDetect yoloPredictor;

        public string DetectImage(string imgPath)
        {
            return DetectImage(imgPath, yoloPredictor);
        }

        public void Dispose()
        {
            Dispose(yoloPredictor);
        }

        public void LoadModel(string modelPath, float confidence, float iou)
        {
            yoloPredictor = new YoloDetect(modelPath, confidence, iou);
        }

        public string SaveImage(FileRowItem item)
        {
            return SaveImage(item, yoloPredictor);
        }
    }
}
