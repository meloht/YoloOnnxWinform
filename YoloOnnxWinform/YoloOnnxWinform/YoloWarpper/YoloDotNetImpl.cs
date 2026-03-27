
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
using YoloDotNet.ExecutionProvider.DirectML;
using YoloDotNet.Extensions;
using YoloDotNet.Models;

namespace YoloOnnxWinform.YoloWarpper
{

    public class YoloDotNetImpl : YoloDetectImplBase, IYoloModel
    {
        private Yolo yoloPredictor;
        private DetectionDrawingOptions _drawingOptions;
        private float _confidence;
        private float _iou;
        public string DetectImage(string imgPath)
        {
            using var image = SKBitmap.Decode(imgPath);
            var data = yoloPredictor.RunObjectDetection(image, confidence: _confidence, iou: this._iou);

            return GetResult(data);
        }

        private string GetResult(List<ObjectDetection> list)
        {
            if (list == null || list.Count == 0)
                return string.Empty;

            var dict = list.GroupBy(p => p.Label.Name).Select(p => $"{p.Count()} {p.Key}").ToList();
            string confs = string.Join(", ", list.Select(p => Math.Round(p.Confidence, 2)));
            return $"{string.Join(", ", dict)} [{confs}]";
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
                ExecutionProvider = new DirectMLExecutionProvider(

                    // Path or byte[] to the ONNX model file.
                    model: modelPath,

                    // GPU device Id to use for inference. -1 = CPU, 0+ = GPU device Id.
                    gpuId: 0

                    // Optional configuration for TensorRT execution.
                    // Executes inference using NVIDIA TensorRT for highly optimized GPU acceleration.
                    // Supports FP32 and FP16 precision modes, and optionally INT8 if calibration data is provided.
                    // trtConfig: new TensorRt {  ... }
                    ),
                ImageResize = ImageResize.Proportional,
                SamplingOptions = new(SKFilterMode.Linear, SKMipmapMode.None) // YoloDotNet default
            });
        }

        public string SaveImage(FileRowItem item)
        {
            using var image = SKBitmap.Decode(item.FilePath);

            // Run object detection inference
            var results = yoloPredictor.RunObjectDetection(image, confidence: _confidence, iou: this._iou);

            // Draw results
            image.Draw(results, _drawingOptions);

            string path = GetSavePath(item);

            // Save image
            image.Save(path, SKEncodedImageFormat.Jpeg, 100);
            return path;
        }
    }

}
