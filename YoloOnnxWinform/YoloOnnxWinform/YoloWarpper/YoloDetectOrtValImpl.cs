using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloOnnxWinform.YoloOnnx;

namespace YoloOnnxWinform.YoloWarpper
{
    public class YoloDetectOrtValImpl : YoloDetectImplBase, IYoloModel
    {
        private YoloDetectOrtVal yoloPredictor;
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
            yoloPredictor = new YoloDetectOrtVal(modelPath, confidence, iou);
        }

        public string SaveImage(FileRowItem item)
        {
            return SaveImage(item, yoloPredictor);
        }
    }
}
