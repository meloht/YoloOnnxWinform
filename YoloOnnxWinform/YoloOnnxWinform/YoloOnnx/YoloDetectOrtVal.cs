using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;
using YoloDotNet.Models;
using static System.Collections.Specialized.BitVector32;


namespace YoloOnnxWinform.YoloOnnx
{
    public class YoloDetectOrtVal : IDisposable
    {
        private readonly float _confidenceThres;
        private readonly float _iouThres;
        private readonly LabelModel[] Labels;
        private readonly string InputName;
        private readonly Scalar[] _colorPalette;
        private readonly Scalar _paddingColor;
        private InferenceSession _session;

        private readonly int _channels;
        private readonly int _channels2;
        private readonly int _channels3;
        private readonly int _channels4;
        private readonly List<Output> _outputs;
        private readonly Input _input;
        private readonly long[] InputShape;

        public int InputWidth { get; private set; }
        public int InputHeight { get; private set; }

        public YoloDetectOrtVal(string onnxModelPath, float confidenceThres, float iouThres)
        {
            _confidenceThres = confidenceThres;
            _iouThres = iouThres;
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            };
            options.EnableCpuMemArena = true;
            _session = new InferenceSession(onnxModelPath, options);
            Labels = MapLabelsAndColors();
            var inputMeta = _session.InputMetadata.First();
            InputName = _session.InputNames[0];

            InputHeight = inputMeta.Value.Dimensions[2];
            InputWidth = inputMeta.Value.Dimensions[3];

            _outputs = GetOutputShapes();
            _input = GetModelInputShape();
            _channels = _outputs[0].Channels;
            _channels2 = _channels * 2;
            _channels3 = _channels * 3;
            _channels4 = _channels * 4;

            var _runOptions = new RunOptions();
            var _ortIoBinding = _session.CreateIoBinding();

            _colorPalette = GenerateColorPalette(Labels.Length);
            _paddingColor = new Scalar(114, 114, 114);

            InputShape = new long[]
              {
                    _session.InputMetadata[InputName].Dimensions[0], // Batch (nr of images the model can process)
                    _session.InputMetadata[InputName].Dimensions[1], // Color channels
                    _session.InputMetadata[InputName].Dimensions[2], // Required image height
                    _session.InputMetadata[InputName].Dimensions[3], // Required image width
              };
        }

        private Scalar[] GenerateColorPalette(int count)
        {
            var rng = new Random();
            var palette = new Scalar[count];
            for (int i = 0; i < count; i++)
            {
                palette[i] = new Scalar(rng.Next(0, 256), rng.Next(0, 256), rng.Next(0, 256));
            }
            return palette;
        }
        private LabelModel[] MapLabelsAndColors()
        {
            var metaData = _session.ModelMetadata.CustomMetadataMap;
            var onnxLabelData = metaData["names"];
            // Labels to Dictionary
            var onnxLabels = onnxLabelData
                .Trim('{', '}')
                .Replace("'", "")
                .Split(", ")
                .Select(x => x.Split(": "))
                .ToDictionary(x => int.Parse(x[0]), x => x[1]);

            return [.. onnxLabels!.Select((label, index) => new LabelModel
            {
                Index = index,
                Name = label.Value,
            })];
        }
        private List<Output> GetOutputShapes()
        {
            var metaData = _session.OutputMetadata;
            var dimensions = metaData.Values.Select(x => x.Dimensions).ToArray();

            var (output0, output1) = (Output.Detection(dimensions[0]), Output.Empty());
            return [output0, output1];
        }
        private Input GetModelInputShape()
        {
            NodeMetadata metaData = _session.InputMetadata[InputName];
            var dimensions = metaData.Dimensions;

            // Check for any dynamic dimension (-1 means dynamic in ONNX)
            if (dimensions.Any(d => d == -1))
                throw new Exception("Dynamic ONNX models are not supported.");

            return Input.Shape(dimensions);
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
            var topLeft = new OpenCvSharp.Point((int)box.X, (int)box.Y);
            var bottomRight = new OpenCvSharp.Point((int)(box.X + box.Width), (int)(box.Y + box.Height));

            double fontScale = 0.7;
            // 绘制边界框
            Cv2.Rectangle(img, topLeft, bottomRight, color, 2);

            // 绘制标签
            string label = $"{Labels[classId].Name}: {score:F2}";
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, fontScale, 1, out int baseline);
            var labelTop = new OpenCvSharp.Point((int)box.X, (int)box.Y - 10);

            if (labelTop.Y < textSize.Height)
                labelTop.Y = (int)box.Y + 10;

            // 标签背景
            Cv2.Rectangle(img,
                new OpenCvSharp.Point(labelTop.X, labelTop.Y - textSize.Height),
                new OpenCvSharp.Point(labelTop.X + textSize.Width, labelTop.Y + baseline),
                color, -1);

            // 标签文本
            Cv2.PutText(img, label, labelTop, HersheyFonts.HersheySimplex, fontScale, Scalar.Black, 1, LineTypes.AntiAlias);
        }

        private (Mat letterboxImg, int top, int left) LetterboxFor1280(Mat img)
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
                value: _paddingColor // 填充色（BGR 格式）
            );

            // 关键检查：确保填充后尺寸严格为 1280×1280
            if (letterboxImg.Rows != InputHeight || letterboxImg.Cols != InputWidth)
            {
                throw new Exception($"Letterbox 后尺寸错误！预期 (1280,1280)，实际 ({letterboxImg.Rows},{letterboxImg.Cols})");
            }
            resizedImg.Dispose();
            return (letterboxImg, padH, padW);
        }

        private List<Detection> Postprocess(Mat inputImage, OrtValue ortTensor, int padTop, int padLeft)
        {
            var ortSpan = ortTensor.GetTensorDataAsSpan<float>();
         
            int imageHeight = inputImage.Height;
            int imageWidth = inputImage.Width;
            List<Rect> boxes = new List<Rect>();
            List<float> scores = new List<float>();
            List<int> class_ids = new List<int>();
            float gain = Math.Min((float)InputHeight / imageHeight, (float)InputWidth / imageWidth);
            for (int i = 0; i < _channels; i++)
            {
                // Move forward to confidence value of first label
                var labelOffset = i + _channels4;

                float bestConfidence = 0f;
                int bestLabelIndex = -1;

                // Get confidence and label for current bounding box
                for (var l = 0; l < Labels.Length; l++, labelOffset += _channels)
                {
                    var boxConfidence = ortSpan[labelOffset];

                    if (boxConfidence > bestConfidence)
                    {
                        bestConfidence = boxConfidence;
                        bestLabelIndex = l;
                    }
                }

                // Stop early if confidence is low
                if (bestConfidence < _confidenceThres)
                    continue;

                float x = ortSpan[i] - padLeft;
                float y = ortSpan[i + _channels] - padTop;
                float w = ortSpan[i + _channels2];
                float h = ortSpan[i + _channels3];

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
                    class_ids.Add(bestLabelIndex);
                    scores.Add(bestConfidence);
                    boxes.Add(new Rect(left, top, width, height));
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
                Rect box = boxes[idx];
                float score = scores[idx];
                int class_id = class_ids[idx];
                string lable = Labels[class_id].Name;

                Detection detection = new Detection();
                detection.Confidence = score;
                detection.ClassName = lable;
                detection.ClassId = class_id;
                detection.Box = box;
                results.Add(detection);

            }

            return results;
        }
        private (float[] data, int top, int left) Preprocess(Mat inputImage)
        {
            // BGR转RGB
            using Mat rgbImg = new Mat();

            Cv2.CvtColor(inputImage, rgbImg, ColorConversionCodes.BGR2RGB);

            // Letterbox处理
            (Mat paddedImg, int top, int left) = LetterboxFor1280(rgbImg);

            // 归一化并转换为float数组
            paddedImg.ConvertTo(paddedImg, MatType.CV_32F, 1.0 / 255.0);

            //// 转换为CHW格式 (3, H, W)
            //var channels = paddedImg.Split();
            //float[] data = new float[3 * paddedImg.Rows * paddedImg.Cols];
            //int index = 0;

            //foreach (var channel in channels)
            //{
            //    float[] channelData = new float[channel.Rows * channel.Cols];
            //    channel.GetArray<float>(out channelData);
            //    Array.Copy(channelData, 0, data, index, channelData.Length);
            //    index += channelData.Length;
            //}

            int channelSize = paddedImg.Height * paddedImg.Width;
            float[] data = ArrayPool<float>.Shared.Rent(3 * channelSize);
            ConvertToCHW(paddedImg, data);

            paddedImg.Dispose();
            // 添加批次维度 (1, 3, H, W)
            return (data, top, left);
        }

        private unsafe void ConvertToCHW(Mat image, float[] data)
        {
            int height = image.Rows;
            int width = image.Cols;
            int channelSize = height * width;


            // 使用指针直接访问，避免Split的开销
            unsafe
            {
                float* ptr = (float*)image.Data;

                fixed (float* dataPtr = data)
                {
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            int srcIndex = (y * width + x) * 3;
                            int dstIndexR = y * width + x;
                            int dstIndexG = dstIndexR + channelSize;
                            int dstIndexB = dstIndexG + channelSize;

                            dataPtr[dstIndexB] = ptr[srcIndex];     // B
                            dataPtr[dstIndexG] = ptr[srcIndex + 1]; // G  
                            dataPtr[dstIndexR] = ptr[srcIndex + 2]; // R
                        }
                    }
                }
            }
        }
        public List<Detection> Run(Mat inputImage)
        {
            // 预处理图像
            (float[] inputData, int top, int left) = Preprocess(inputImage);

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputData, InputShape);

            using var runOptions = new RunOptions();
            // 执行推理
            using var outputs = _session.Run(runOptions, [InputName], [inputOrtValue], _session.OutputNames);
            using var output_0 = outputs[0];

            ArrayPool<float>.Shared.Return(inputData);
            // 后处理
            var result = Postprocess(inputImage, output_0, top, left);


            return result;
        }

        public void Dispose()
        {
            _session.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
