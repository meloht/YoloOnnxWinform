
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloDotNet;
using YoloDotNet.Core;
using YoloDotNet.Enums;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

namespace YoloOnnxWinform.YoloWarpper
{

    public class YoloDotNetImpl : IYoloModel
    {
        private Yolo yoloPredictor;
        private DetectionDrawingOptions _drawingOptions;
        private float _confidence;
        private float _iou;
        public DataModel DetectImage(string imgPath)
        {
            DataModel model = new DataModel();

            using var image = SKBitmap.Decode(imgPath);
            var data = yoloPredictor.RunObjectDetection(image, confidence: _confidence, iou: this._iou);

            model.DetectionResult = data.Count.ToString();

            return model;
        }

        public void Dispose()
        {
            yoloPredictor?.Dispose();
        }

        public void LoadModel(string modelPath, float confidence, float iou)
        {
            _drawingOptions = new DetectionDrawingOptions
            {
                DrawBoundingBoxes = true,
                DrawConfidenceScore = true,
                DrawLabels = true,
                EnableFontShadow = true,

                Font = SKTypeface.Default,

                FontSize = 18,
                FontColor = SKColors.Blue,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
                BorderThickness = 2,

                BoundingBoxOpacity = 64,

            };
            this._confidence = confidence;
            this._iou = iou;
            yoloPredictor = new Yolo(new YoloOptions
            {
                OnnxModel = modelPath,
                ExecutionProvider = new CpuExecutionProvider(),
                ImageResize = ImageResize.Proportional,
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default
            });
        }

        public string SaveImage(FileRowItem item)
        {
            using var image = SKBitmap.Decode(item.FilePath);

            // Run object detection inference
            var results = yoloPredictor.RunObjectDetection(image, confidence: _confidence, iou: this._iou);

            // Draw results
            image.Draw(results, _drawingOptions);

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

            // Save image
            image.Save(path, SKEncodedImageFormat.Jpeg, 100);
            return path;
        }
    }
}
