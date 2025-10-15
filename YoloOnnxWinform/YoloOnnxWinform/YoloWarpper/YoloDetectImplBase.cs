using Compunet.YoloSharp;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloOnnxWinform.YoloOnnx;

namespace YoloOnnxWinform.YoloWarpper
{
    public class YoloDetectImplBase
    {
        protected DataModel DetectImage(string imgPath, IYoloDetect yoloPredictor)
        {
            DataModel model = new DataModel();

            using Mat inputImage = Cv2.ImRead(imgPath);
            var data = yoloPredictor.Run(inputImage);

            model.DetectionResult = GetResult(data);

            return model;
        }
        private string GetResult(List<Detection> list)
        {
            if (list == null || list.Count == 0)
                return string.Empty;

            var dict = list.GroupBy(p => p.ClassName).Select(p => $"{p.Count()} {p.Key}").ToList();
            return string.Join(", ", dict);
        }
        protected void Dispose(IYoloDetect yoloPredictor)
        {
            yoloPredictor?.Dispose();
        }
        protected string SaveImage(FileRowItem item, IYoloDetect yoloPredictor)
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
