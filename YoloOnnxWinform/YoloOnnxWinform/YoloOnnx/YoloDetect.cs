
using Compunet.YoloSharp;
using Compunet.YoloSharp.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using SharpDX;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Collections.Specialized.BitVector32;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace YoloOnnxWinform.YoloOnnx
{
    public class YoloDetect : IDisposable
    {
        public readonly string _onnxModelPath;
        private readonly float _confidenceThres;
        private readonly float _iouThres;
        private readonly List<string> _classes;
        private readonly Scalar[] _colorPalette;
        public int InputWidth { get; private set; }
        public int InputHeight { get; private set; }
        private InferenceSession session;
        private bool disposedValue;
        private static readonly List<string> CocoClasses = ["blackdot"];


        public YoloDetect(string onnxModelPath, float confidenceThres, float iouThres)
        {
            _onnxModelPath = onnxModelPath;
            _confidenceThres = confidenceThres;
            _iouThres = iouThres;
            _classes = CocoClasses;
            _colorPalette = GenerateColorPalette(_classes.Count);
            session = new InferenceSession(onnxModelPath);
        }

        private Scalar[] GenerateColorPalette(int count)
        {
            var rng = new Random();
            return Enumerable.Range(0, count)
                .Select(_ => new Scalar(rng.Next(0, 256), rng.Next(0, 256), rng.Next(0, 256)))
                .ToArray();
        }
        public void DrawDetections(Mat inputImage, List<Detection> list)
        {
            foreach (var item in list)
            {
                DrawDetections(inputImage, item.Box, item.Confidence, item.ClassId);
            }
        }
        public void DrawDetections(Mat img, Rect box, float score, int classId)
        {
            var color = _colorPalette[classId];
            var topLeft = new OpenCvSharp.Point(box.X, box.Y);
            var bottomRight = new OpenCvSharp.Point(box.X + box.Width, box.Y + box.Height);

            double fontScale = 0.7;
            // 绘制边界框
            Cv2.Rectangle(img, topLeft, bottomRight, color, 2);

            // 绘制标签
            string label = $"{_classes[classId]}: {score:F2}";
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, fontScale, 1, out int baseline);
            var labelTop = new OpenCvSharp.Point(box.X, box.Y - 10);

            if (labelTop.Y < textSize.Height)
                labelTop.Y = box.Y + 10;

            // 标签背景
            Cv2.Rectangle(img,
                new OpenCvSharp.Point(labelTop.X, labelTop.Y - textSize.Height),
                new OpenCvSharp.Point(labelTop.X + textSize.Width, labelTop.Y + baseline),
                color, -1);

            // 标签文本
            Cv2.PutText(img, label, labelTop, HersheyFonts.HersheySimplex, fontScale, Scalar.Black, 1, LineTypes.AntiAlias);
        }
        private (Mat letterboxImg, int topPad, int leftPad) LetterboxFor1280(Mat img)
        {
            // 1. 获取原始图像尺寸
            int imgH = img.Rows;
            int imgW = img.Cols;

            // 2. 计算缩放比例（按最小比例缩放，避免图像畸变）
            float scale = Math.Min((float)InputHeight / imgH, (float)InputWidth / imgW);

            // 3. 计算缩放后的尺寸（确保按比例缩放）
            int newImgW = (int)Math.Round(imgW * scale);
            int newImgH = (int)Math.Round(imgH * scale);

            // 4. 计算填充值（左右填充、上下填充，确保最终尺寸=1280×1280）
            int padW = (InputWidth - newImgW) / 2; // 左右填充的一半
            int padH = (InputHeight - newImgH) / 2; // 上下填充的一半

            // 5. 缩放图像（若原始尺寸≠缩放后尺寸）
            Mat resizedImg = new Mat();
            if (imgW != newImgW || imgH != newImgH)
            {
                Cv2.Resize(img, resizedImg, new OpenCvSharp.Size(newImgW, newImgH), interpolation: InterpolationFlags.Linear);
            }
            else
            {
                resizedImg = img.Clone();
            }

            // 6. 填充到 1280×1280（用 114 填充，YOLO 常用默认值）
            Mat letterboxImg = new Mat();
            Cv2.CopyMakeBorder(
                src: resizedImg,
                dst: letterboxImg,
                top: padH,        // 顶部填充
                bottom: InputHeight - newImgH - padH, // 底部填充（补全到 1280）
                left: padW,       // 左侧填充
                right: InputWidth - newImgW - padW,  // 右侧填充（补全到 1280）
                borderType: BorderTypes.Constant,
                value: new Scalar(114, 114, 114) // 填充色（BGR 格式）
            );
            resizedImg.Dispose();
            // 关键检查：确保填充后尺寸严格为 1280×1280
            if (letterboxImg.Rows != InputHeight || letterboxImg.Cols != InputWidth)
            {
                throw new Exception($"Letterbox 后尺寸错误！预期 (1280,1280)，实际 ({letterboxImg.Rows},{letterboxImg.Cols})");
            }

            return (letterboxImg, padH, padW);
        }
        private (float[] data, int topPad, int leftPad) Preprocess(Mat inputImage, int imageHeight, int imageWidth)
        {
            // BGR转RGB
            using Mat rgbImg = new Mat();

            Cv2.CvtColor(inputImage, rgbImg, ColorConversionCodes.BGR2RGB);

            // Letterbox处理
            (Mat paddedImg, int topPad, int leftPad) = LetterboxFor1280(rgbImg);

            // 归一化并转换为float数组
            paddedImg.ConvertTo(paddedImg, MatType.CV_32F, 1.0 / 255.0);

            // 转换为CHW格式 (3, H, W)
            var channels = paddedImg.Split();
            float[] data = new float[3 * paddedImg.Rows * paddedImg.Cols];
            int index = 0;

            foreach (var channel in channels)
            {
                float[] channelData = new float[channel.Rows * channel.Cols];
                channel.GetArray(out channelData);
                Array.Copy(channelData, 0, data, index, channelData.Length);
                index += channelData.Length;
            }
            foreach (var item in channels)
            {
                item.Dispose();
            }
            paddedImg.Dispose();
            // 添加批次维度 (1, 3, H, W)
            return (data, topPad, leftPad);
        }

        private float[,] ProcessTensorOutput(Tensor<float> outputTensor)
        {
            // 获取Tensor的维度
            var dimensions = outputTensor.Dimensions.ToArray();

            // YOLOv8输出通常是 [1, 84, 8400] 格式
            // 我们需要转置并挤压，变成 [8400, 84]

            if (dimensions.Length == 3 && dimensions[0] == 1)
            {
                // 挤压第一维 (batch size = 1)
                int features = dimensions[1];  // 84
                int detections = dimensions[2]; // 8400

                // 创建新的二维数组 [detections, features]
                float[,] result = new float[detections, features];

                // 填充数据 (相当于转置)
                for (int i = 0; i < detections; i++)
                {
                    for (int j = 0; j < features; j++)
                    {
                        result[i, j] = outputTensor[0, j, i];
                    }
                }

                return result;
            }
            else if (dimensions.Length == 2)
            {
                // 如果已经是2D，直接转换为二维数组
                int rows = dimensions[0];
                int cols = dimensions[1];
                float[,] result = new float[rows, cols];

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        result[i, j] = outputTensor[i, j];
                    }
                }

                return result;
            }
            else
            {
                throw new Exception($"Unexpected tensor dimensions: [{string.Join(", ", dimensions)}]");
            }
        }

        private List<Detection> Postprocess(Mat inputImage, Tensor<float> outputTensor, int topPad, int leftPad)
        {
            int imageHeight = inputImage.Rows;
            int imageWidth = inputImage.Cols;
            // Transpose and squeeze the output to match the expected shape
            var processedOutput = ProcessTensorOutput(outputTensor);

            // Get the number of rows in the outputs array
            int rows = processedOutput.GetLength(0);
            // Lists to store the bounding boxes, scores, and class IDs of the detections
            List<Rect> boxes = new List<Rect>();
            List<float> scores = new List<float>();
            List<int> class_ids = new List<int>();

            // Calculate the scaling factors for the bounding box coordinates
            float gain = Math.Min((float)InputHeight / imageHeight, (float)InputWidth / imageWidth);

            // Iterate over each row in the outputs array
            for (int i = 0; i < rows; i++)
            {
                // Extract the class scores from the current row
                float max_score = 0;
                int class_id = -1;

                // Find the maximum score among the class scores
                for (int j = 4; j < processedOutput.GetLength(1); j++)
                {
                    float score = processedOutput[i, j];
                    if (score > max_score)
                    {
                        max_score = score;
                        class_id = j - 4;
                    }
                }

                // If the maximum score is above the confidence threshold
                if (max_score >= _confidenceThres && class_id != -1)
                {
                    // Extract the bounding box coordinates from the current row
                    // Adjust for padding
                    float x = processedOutput[i, 0] - leftPad;  // x_center
                    float y = processedOutput[i, 1] - topPad;  // y_center
                    float w = processedOutput[i, 2];              // width
                    float h = processedOutput[i, 3];              // height

                    // Calculate the scaled coordinates of the bounding box
                    int left = (int)((x - w / 2) / gain);
                    int top = (int)((y - h / 2) / gain);
                    int width = (int)(w / gain);
                    int height = (int)(h / gain);

                    // Ensure coordinates are within image bounds
                    left = Math.Max(0, left);
                    top = Math.Max(0, top);
                    width = Math.Min(width, imageWidth - left);
                    height = Math.Min(height, imageHeight - top);

                    // Add the class ID, score, and box coordinates to the respective lists
                    if (width > 0 && height > 0)
                    {
                        class_ids.Add(class_id);
                        scores.Add(max_score);
                        boxes.Add(new Rect(left, top, width, height));
                    }
                }
            }

            // 非极大值抑制
            int[] indices = [];
            if (boxes.Count > 0)
            {
                CvDnn.NMSBoxes(boxes, scores, _confidenceThres, _iouThres, out indices);
            }
            List<Detection> results = new List<Detection>();
            // 绘制检测结果
            foreach (var idx in indices)
            {
                Detection detection = new Detection();
                detection.Confidence = scores[idx];
                detection.ClassId = class_ids[idx];
                detection.ClassName = _classes[detection.ClassId];
                detection.Box = boxes[idx];
                results.Add(detection);
            }

            return results;
        }

        public List<Detection> Run(Mat inputImage)
        {

            var inputMeta = session.InputMetadata.First().Value;
            var inputDims = inputMeta.Dimensions;


            InputHeight = inputDims[2];
            InputWidth = inputDims[3];

            int imageHeight = inputImage.Rows;
            int imageWidth = inputImage.Cols;

            // 预处理图像
            (float[] inputData, int topPad, int leftPad) = Preprocess(inputImage, imageHeight, imageWidth);
            string inputName = session.InputNames[0];
            // 准备输入
            var inputTensor = new DenseTensor<float>(inputData, inputDims);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            // 执行推理
            using var outputs = session.Run(inputs);
            var outputTensor = outputs[0].AsTensor<float>();

            // 后处理
            var result = Postprocess(inputImage, outputTensor, topPad, leftPad);


            return result;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                session.Dispose();
                session = null;
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloDetect()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
