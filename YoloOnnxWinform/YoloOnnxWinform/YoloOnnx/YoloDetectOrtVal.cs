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




namespace YoloOnnxWinform.YoloOnnx
{
    public class YoloDetectOrtVal : YoloDetectBase, IYoloDetect
    {
        private readonly float _confidenceThres;
        private readonly float _iouThres;
        private readonly LabelModel[] Labels;
        private readonly string InputName;
      
        private InferenceSession _session;

        private readonly int _channels;
        private readonly int _channels2;
        private readonly int _channels3;
        private readonly int _channels4;
        private readonly List<Output> _outputs;
        private readonly Input _input;
        private readonly long[] InputShape;


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
            Labels = MapLabelsAndColors(_session);
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
           

            InputShape = new long[]
              {
                    _session.InputMetadata[InputName].Dimensions[0], // Batch (nr of images the model can process)
                    _session.InputMetadata[InputName].Dimensions[1], // Color channels
                    _session.InputMetadata[InputName].Dimensions[2], // Required image height
                    _session.InputMetadata[InputName].Dimensions[3], // Required image width
              };

            RentDataInt(_session.InputMetadata[InputName].Dimensions);
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
            var channels = paddedImg.Split();
  
            float[] data = base.rentData;
          
            int index = 0;

            foreach (var channel in channels)
            {
                float[] channelData = new float[channel.Rows * channel.Cols];
                channel.GetArray<float>(out channelData);
                Array.Copy(channelData, 0, data, index, channelData.Length);
                index += channelData.Length;
            }
            foreach (var item in channels)
            {
                item.Dispose();
            }
            //int channelSize = paddedImg.Height * paddedImg.Width;
            //float[] data = ArrayPool<float>.Shared.Rent(3 * channelSize);
            //ConvertToCHW(paddedImg, data);

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

            //ArrayPool<float>.Shared.Return(inputData);
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
