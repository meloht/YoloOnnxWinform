using Compunet.YoloSharp;
using Compunet.YoloSharp.Data;
using Compunet.YoloSharp.Plotting;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform.YoloWarpper
{
    public class YoloSharpImpl : IYoloModel
    {
        private YoloPredictor _yoloPredictor;

        public DataModel DetectImage(string imgPath)
        {
            DataModel model = new DataModel();

            var data = _yoloPredictor.Detect(imgPath);
            model.DetectionResult = data.ToString();

            return model;
        }

        public void Dispose()
        {
            _yoloPredictor?.Dispose();
        }

        public void LoadModel(string modelPath, float confidence, float iou)
        {
            _yoloPredictor = new YoloPredictor(modelPath);
            _yoloPredictor.Configuration.SuppressParallelInference = true;
            _yoloPredictor.Configuration.KeepAspectRatio = true;
            _yoloPredictor.Configuration.Confidence = confidence;
            _yoloPredictor.Configuration.IoU = iou;
        }

        public string SaveImage(FileRowItem item)
        {
            var result = _yoloPredictor.Detect(item.FilePath);
            using var image = SixLabors.ImageSharp.Image.Load(item.FilePath);

            using var plot = result.PlotImage(image);
            if (plot != null)
            {
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
                plot.Save(path);
                return path;
            }
            return null;
        }
    }
}
