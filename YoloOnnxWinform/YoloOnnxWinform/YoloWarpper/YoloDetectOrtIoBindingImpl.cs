using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using YoloOnnx;
using YoloOnnxWinform;
using YoloOnnxWinform.YoloOnnx;
using YoloOnnxWinform.YoloWarpper;

namespace YoloWarpper
{
    public class YoloDetectOrtIoBindingImpl : YoloDetectImplBase, IYoloModel, IYoloParallel
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
            yoloPredictor = YoloDetectFactory.CreateYoloDetect(modelPath, confidence, iou, YoloWarpperType.YoloDetectOrtIoBind);
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
