using Compunet.YoloSharp;
using Compunet.YoloSharp.Data;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using YoloOnnxWinform;
using YoloOnnxWinform.YoloWarpper;
using YoloSharpOnnx;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;

namespace YoloWarpper
{
    public class YoloSharpOnnxImpl : YoloDetectImplBase, IYoloModel
    {
        public YoloSharp YoloSharp { get; set; }
        public string DetectImage(string imgPath)
        {
            using Mat inputImage = Cv2.ImRead(imgPath);
            var data = YoloSharp.RunDetect(inputImage);
            return data.Summary();
        }

        public async Task<string> DetectImageAsync(string imgPath, IYoloAsync yoloAsync)
        {
            using Mat inputImage = Cv2.ImRead(imgPath);
            var data =await yoloAsync.RunDetectAsync(inputImage);
            return data.Summary();
        }


        public void Dispose()
        {
            YoloSharp?.Dispose();
        }

        public void LoadModel(string modelPath, float confidence, float iou, int deviceId)
        {
            YoloSharp = new YoloSharp(confidence, iou, new ExecutionProviderDirectML(modelPath, deviceId));

        }

        public string SaveImage(FileRowItem item)
        {
            using Mat inputImage = Cv2.ImRead(item.FilePath);

            var result = YoloSharp.RunDetect(inputImage);
            YoloSharp.DrawDetections(inputImage, result);

            return SaveImagePath(inputImage, item);
        }




    }
}
