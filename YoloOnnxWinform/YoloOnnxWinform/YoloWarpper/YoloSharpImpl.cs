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
    public class YoloSharpImpl : YoloDetectImplBase, IYoloModel
    {
        private YoloPredictor _yoloPredictor;

        public string DetectImage(string imgPath)
        {
            var data = _yoloPredictor.Detect(imgPath);
            var dict = data.GroupBy(p => p.Name).Select(p => $"{p.Count()} {p.Key}").ToList();
            string confs = string.Join(", ", data.Select(p => Math.Round(p.Confidence, 2)));
            return $"{string.Join(", ", dict)} [{confs}]";
        }

        public void Dispose()
        {
            _yoloPredictor?.Dispose();
        }

        public void LoadModel(string modelPath, float confidence, float iou)
        {
            _yoloPredictor = new YoloPredictor(modelPath);
            _yoloPredictor.Configuration.Confidence = confidence;
            _yoloPredictor.Configuration.IoU = iou;
            _yoloPredictor.Configuration.KeepAspectRatio = true;
            _yoloPredictor.Configuration.ApplyAutoOrient = true;

        }

        public string SaveImage(FileRowItem item)
        {
            var result = _yoloPredictor.Detect(item.FilePath);
            using var image = SixLabors.ImageSharp.Image.Load(item.FilePath);

            using var plot = result.PlotImage(image);
            if (plot != null)
            {
                string path = GetSavePath(item);
                plot.Save(path);
                return path;
            }
            return null;
        }
    }
}
