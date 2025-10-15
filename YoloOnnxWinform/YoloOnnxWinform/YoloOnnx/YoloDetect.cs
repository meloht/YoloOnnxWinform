

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace YoloOnnxWinform.YoloOnnx
{
    public class YoloDetect : YoloDetectBase, IYoloDetect
    {
        public readonly string _onnxModelPath;
        private readonly float _confidenceThres;
        private readonly float _iouThres;
        private readonly LabelModel[] Labels;

        private InferenceSession session;
        private bool disposedValue;

        public YoloDetect(string onnxModelPath, float confidenceThres, float iouThres)
        {
            _onnxModelPath = onnxModelPath;
            _confidenceThres = confidenceThres;
            _iouThres = iouThres;
            session = new InferenceSession(onnxModelPath);
            Labels = MapLabelsAndColors(session);
            _colorPalette = GenerateColorPalette(Labels.Length);
            var inputMeta = session.InputMetadata.First().Value;
            var inputDims = inputMeta.Dimensions;
            InputHeight = inputDims[2];
            InputWidth = inputDims[3];
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

        private List<Detection> Postprocess(Mat inputImage, Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> outputTensor, int topPad, int leftPad)
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
                detection.ClassName = this.Labels[detection.ClassId].Name;
                detection.Box = boxes[idx];
                results.Add(detection);
            }

            return results;
        }

        public List<Detection> Run(Mat inputImage)
        {
            var inputMeta = session.InputMetadata.First().Value;
            var inputDims = inputMeta.Dimensions;

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
