using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.ComponentModel;
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
        private SessionOptions _options;

        private readonly int _channels;
        private readonly int _channels2;
        private readonly int _channels3;
        private readonly int _channels4;
        private readonly List<Output> _outputs;
        private readonly Input _input;
        private readonly long[] InputShape;


        public YoloDetectOrtVal(InferenceSession session, SessionOptions options, float confidenceThres, float iouThres)
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

            _outputs = GetOutputShapes();
            _input = GetModelInputShape();
            _channels = _outputs[0].Channels;
            _channels2 = _channels * 2;
            _channels3 = _channels * 3;
            _channels4 = _channels * 4;


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

        private List<Detection> Postprocess(int imageHeight, int imageWidth, ReadOnlySpan<float> ortSpan, int padTop, int padLeft)
        {
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



        public List<Detection> Run(Mat inputImage)
        {
            // 预处理图像
            var imgData = Preprocess(inputImage);

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(imgData.OutData, InputShape);

            using var runOptions = new RunOptions();
            // 执行推理
            using var outputs = _session.Run(runOptions, [InputName], [inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            // 后处理
            var result = Postprocess(inputImage.Height, inputImage.Width, output0.GetTensorDataAsSpan<float>(), imgData.TopPad, imgData.LeftPad);


            return result;
        }

        public void Dispose()
        {
            StopLoad();
            _session.Dispose();
            _options.Dispose();
            GC.SuppressFinalize(this);
        }

        public void Run(ImagePreprocessModel model)
        {
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(model.Data, InputShape);

            using var runOptions = new RunOptions();
            // 执行推理
            using var outputs = _session.Run(runOptions, [InputName], [inputOrtValue], _session.OutputNames);
            ArrayPool<float>.Shared.Return(model.Data);
            using var output0 = outputs[0];
            Postprocess(output0.GetTensorDataAsSpan<float>(), model);
        }

        public void Postprocess(ReadOnlySpan<float> ortTensor, ImagePreprocessModel imageData)
        {
            var list = Postprocess(imageData.imageHeight, imageData.imageWidth, ortTensor, imageData.TopPad, imageData.LeftPad);
            imageData.model.DetectionResult = Utils.GetResult(list);

        }

        public void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict)
        {
            base._listName = list;
            base._dict = dict;
            Start();
        }

        public ImagePreprocessModel[] GetPreLoadImages()
        {
            return GetPreImgs();
        }

        public void EndPreload()
        {
            StopLoad();
        }
    }
}
