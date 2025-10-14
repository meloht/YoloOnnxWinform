using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform.YoloWarpper
{
    public class YoloDetectImpl : IYoloModel
    {
        private YoloDetect yoloPredictor;
        public DataModel DetectImage(string imgPath)
        {
            DataModel model = new DataModel();

            using Mat inputImage = Cv2.ImRead(imgPath);
            var data = yoloPredictor.Run(inputImage);
        
            model.DetectionResult = data.Count.ToString();
          
            return model;
        }

        public void Dispose()
        {
            yoloPredictor?.Dispose();
        }

        public void LoadModel(string modelPath, float confidence, float iou)
        {
            yoloPredictor = new YoloDetect(modelPath, confidence, iou);
        }

        public string SaveImage(FileRowItem item)
        {
            using Mat inputImage = Cv2.ImRead(item.FilePath);
            var result = yoloPredictor.Run(inputImage);
            yoloPredictor.DrawDetections(inputImage, result);
            string folder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "temp");
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }
            string path = Path.Combine(folder, item.FileName);
            if (File.Exists(path))
            {
                File.Delete(path);
            }
            Cv2.ImWrite(path, inputImage);
            return path;
        }
    }
}
