using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using YoloOnnxWinform;
using YoloOnnxWinform.YoloOnnx;
using static System.Net.Mime.MediaTypeNames;

namespace YoloOnnx
{
    public class YoloDetectEndToEndOrtVal : YoloDetectBase, IYoloDetect
    {
        private readonly float _confidenceThres;
        private readonly float _iouThres;
        private readonly LabelModel[] Labels;
        private readonly string InputName;

        private InferenceSession _session;
        private SessionOptions _options;

        private readonly int _channels;

        private readonly long[] InputShape;

        public YoloDetectEndToEndOrtVal(InferenceSession session, SessionOptions options, float confidenceThres, float iouThres)
        {
            _confidenceThres = confidenceThres;
            _iouThres = iouThres;

            _options = options;
            _session = session;
            Labels = MapLabelsAndColors(_session);
            var inputMeta = _session.InputMetadata.First();
            InputName = _session.InputNames[0];

            InputHeight = inputMeta.Value.Dimensions[2];
            InputWidth = inputMeta.Value.Dimensions[3];

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

        public void Dispose()
        {
            StopLoad();
            _session.Dispose();
            _options.Dispose();
            _resizedImg.Dispose();
            GC.SuppressFinalize(this);
        }

        public void EndPreload()
        {
            StopLoad();
        }

        public ImagePreprocessModel[] GetPreLoadImages()
        {
            return GetPreImgs();
        }

        public void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict)
        {
            base._listName = list;
            base._dict = dict;
            Start();
        }
        private void Preprocess(Mat image, float ratio, float[] data)
        {
            // 1. Preprocessing (Letterbox)
            int newWidth = (int)(image.Width * ratio);
            int newHeight = (int)(image.Height * ratio);
            using Mat rgbImg = new Mat();

            Cv2.CvtColor(image, rgbImg, ColorConversionCodes.BGR2RGB);
            using var resized = new Mat();
            Cv2.Resize(rgbImg, resized, new OpenCvSharp.Size(newWidth, newHeight));

            using var canvas = new Mat(new OpenCvSharp.Size(InputWidth, InputHeight), MatType.CV_8UC3, new Scalar(114, 114, 114));
            resized.CopyTo(new Mat(canvas, new Rect(0, 0, newWidth, newHeight)));

            // 2. 归一化并转换为 Tensor (HWC -> CHW)
            GetChwArr(canvas, data);
        }
        public List<Detection> Run(Mat inputImage)
        {
            // 1. Preprocessing (Letterbox)
            float[] data = _inputBuffer;
           
            var preRes = Preprocess(inputImage, data);
            // 3. 推理
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(data, InputShape);
            using var runOptions = new RunOptions();

            using var results = _session.Run(runOptions, [InputName], [inputOrtValue], _session.OutputNames);
            using var output0 = results[0];

            // 4. 后处理 (YOLO26 直接输出 [1, 300, 6])
            return PostProcess(output0, _confidenceThres, preRes.Scale, preRes.LeftPad, preRes.TopPad);
        }

        public void Run(ImagePreprocessModel model)
        {

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(model.Data, InputShape);

            using var runOptions = new RunOptions();
            // 执行推理
            using var outputs = _session.Run(runOptions, [InputName], [inputOrtValue], _session.OutputNames);
            ArrayPool<float>.Shared.Return(model.Data);
            using var output0 = outputs[0];
            PostprocessModel(output0, model);
        }
        public void PostprocessModel(OrtValue ortTensor, ImagePreprocessModel imageData)
        {
            var list = PostProcess(ortTensor, _confidenceThres, imageData.Scale, imageData.PadX, imageData.PadY);
            imageData.model.DetectionResult = Utils.GetResult(list);
        }


        public List<Detection> PostProcess(OrtValue outputValue, float threshold, float scale, int padx, int pady)
        {
            var detections = new List<Detection>();

            // 1. 获取第一个输出张量
            var shape = outputValue.GetTensorTypeAndShape().Shape; // 例如 [1, 300, 6]

            int rowCount = (int)shape[1]; // 300
            int colCount = (int)shape[2]; // 6

            // 2. 使用 Span 直接访问内存，避免产生垃圾回收
            ReadOnlySpan<float> data = outputValue.GetTensorDataAsSpan<float>();

            for (int i = 0; i < rowCount; i++)
            {
                // 计算当前行的偏移量
                int offset = i * colCount;

                float confidence = data[offset + 4];

                // 过滤低置信度结果
                if (confidence < threshold) continue;

                // 3. 提取坐标并还原到原始图像尺寸
                // 注意：YOLOv26 默认输出通常是 [x1, y1, x2, y2]
                float x1 = (data[offset + 0] - padx) / scale;
                float y1 = (data[offset + 1] - pady) / scale;
                float x2 = (data[offset + 2] - padx) / scale;
                float y2 = (data[offset + 3] - pady) / scale;

                int labelId = (int)data[offset + 5];

                detections.Add(new Detection
                {
                    Box = new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1)),
                    Confidence = confidence,
                    ClassId = labelId,
                    ClassName = Labels[labelId].Name
                });
            }

            return detections;
        }

    }
}
