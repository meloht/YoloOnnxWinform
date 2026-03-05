using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloOnnx;
using YoloOnnxWinform.YoloOnnx;

namespace YoloOnnxWinform.YoloWarpper
{
    public class YoloDetectImpl : YoloDetectImplBase, IYoloModel, IYoloParallel
    {
        private IYoloDetect yoloPredictor;

        public string DetectImage(string imgPath)
        {
            return DetectImage(imgPath, yoloPredictor);
        }

        public void Dispose()
        {
            Dispose(yoloPredictor);
        }


        public ImagePreprocessModel[] GetPreLoadImages()
        {
            return yoloPredictor.GetPreLoadImages();
        }

        public void LoadModel(string modelPath, float confidence, float iou)
        {
            yoloPredictor = YoloDetectFactory.CreateYoloDetect(modelPath, confidence, iou, YoloWarpperType.YoloDetect);
        }



        public void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict)
        {
            yoloPredictor.PreLoadImages(list, dict);
        }

        public void Run(ImagePreprocessModel model)
        {
            yoloPredictor.Run(model);
        }

        public string SaveImage(FileRowItem item)
        {
            return SaveImage(item, yoloPredictor);
        }

        public void EndPreload()
        {
            yoloPredictor.EndPreload();
        }
    }
}
