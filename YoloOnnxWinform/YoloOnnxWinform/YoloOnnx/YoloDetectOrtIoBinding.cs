using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using YoloOnnxWinform;
using YoloOnnxWinform.YoloOnnx;

namespace YoloOnnx
{
    public class YoloDetectOrtIoBinding : YoloDetectBase, IYoloDetect
    {
        private readonly float _confidenceThres;
        private readonly float _iouThres;
        private readonly LabelModel[] Labels;


        private readonly string _inputName;
        private readonly string _outputName;

        private InferenceSession _session;
        private SessionOptions _options;
        private readonly RunOptions _runOptions;

        private readonly long[] _inputShape;
        private readonly long _inputSizeInBytes;
        private readonly long[] _outputShape;
        private readonly long _outputSizeInBytes;

        private readonly OrtSafeMemoryHandle _inputNativeAllocation;
        private readonly OrtSafeMemoryHandle _outputNativeAllocation;

        private readonly int _boxNums;
        private readonly int _boxNums2;
        private readonly int _boxNums3;
        private readonly int _boxNums4;




        public YoloDetectOrtIoBinding(InferenceSession session, SessionOptions options, float confidenceThres, float iouThres)
        {
            _confidenceThres = confidenceThres;
            _iouThres = iouThres;

            _options = options;
            _session = session;
            Labels = MapLabelsAndColors(_session);
            var inputMeta = _session.InputMetadata.First();
            _inputName = _session.InputNames[0];
            _outputName = _session.OutputNames[0];

            InputHeight = inputMeta.Value.Dimensions[2];
            InputWidth = inputMeta.Value.Dimensions[3];

            _runOptions = new RunOptions();

            var inputMetaData = _session.InputMetadata;
            var outputMetaData = _session.OutputMetadata;

            _inputShape = Array.ConvertAll<int, long>(inputMetaData[_inputName].Dimensions, Convert.ToInt64);
            _outputShape = Array.ConvertAll<int, long>(outputMetaData[_outputName].Dimensions, Convert.ToInt64);


            _boxNums = outputMetaData[_outputName].Dimensions[2];
            _boxNums2 = _boxNums * 2;
            _boxNums3 = _boxNums * 3;
            _boxNums4 = _boxNums * 4;

            _colorPalette = GenerateColorPalette(Labels.Length);

            var inputShapeSize = ShapeUtils.GetSizeForShape(_inputShape);
            var outputShapeSize = ShapeUtils.GetSizeForShape(_outputShape);


            _inputSizeInBytes = inputShapeSize * sizeof(float);
            IntPtr allocPtr = Marshal.AllocHGlobal((int)_inputSizeInBytes);
            _inputNativeAllocation = new OrtSafeMemoryHandle(allocPtr);

            _outputSizeInBytes = outputShapeSize * sizeof(float);
            allocPtr = Marshal.AllocHGlobal((int)_outputSizeInBytes);
            _outputNativeAllocation = new OrtSafeMemoryHandle(allocPtr);


            RentDataInt(_session.InputMetadata[_inputName].Dimensions);
        }





        public void Dispose()
        {
            StopLoad();
            _session.Dispose();
            _options.Dispose();
            _runOptions.Dispose();
            _inputNativeAllocation.Dispose();
            _outputNativeAllocation.Dispose();
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

        public List<Detection> Run(Mat inputImage)
        {
            ClearOutput();
            ClearInput();
            var imgData = Preprocess(inputImage, _inputBuffer);
            PopulateNativeBuffer<float>(_inputNativeAllocation.Handle, imgData.OutData);


            using (var ioBinding = _session.CreateIoBinding())
            {

                using (var tensor = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance,
                      TensorElementType.Float,
                      _inputShape, _inputNativeAllocation.Handle, _inputSizeInBytes))
                {
                    ioBinding.BindInput(_inputName, tensor);
                }

                // The output will go into the Ort allocated OrtValue
                ioBinding.BindOutputToDevice(_outputName, OrtMemoryInfo.DefaultInstance);
                ioBinding.SynchronizeBoundInputs();


                using (var results = _session.RunWithBoundResults(_runOptions, ioBinding))
                {
                    ioBinding.SynchronizeBoundOutputs();

                    using var output0 = results[0];

                    // 后处理
                    var result = Postprocess(inputImage.Height, inputImage.Width, output0.GetTensorDataAsSpan<float>(), imgData.TopPad, imgData.LeftPad);

                    return result;

                }
            }




        }

        public void Run(ImagePreprocessModel model)
        {

            ClearOutput();
            ClearInput();

            PopulateNativeBuffer<float>(_inputNativeAllocation.Handle, model.Data);


            using (var ioBinding = _session.CreateIoBinding())
            {

                using (var tensor = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance,
                      TensorElementType.Float,
                      _inputShape, _inputNativeAllocation.Handle, _inputSizeInBytes))
                {
                    ioBinding.BindInput(_inputName, tensor);
                }

                // The output will go into the Ort allocated OrtValue
                ioBinding.BindOutputToDevice(_outputName, OrtMemoryInfo.DefaultInstance);
                ioBinding.SynchronizeBoundInputs();


                using (var results = _session.RunWithBoundResults(_runOptions, ioBinding))
                {
                    ioBinding.SynchronizeBoundOutputs();

                    using var output0 = results[0];
                    ArrayPool<float>.Shared.Return(model.Data);

                    Postprocess(output0.GetTensorDataAsSpan<float>(), model);


                }

            }
        }

        public void Postprocess(ReadOnlySpan<float> ortTensor, ImagePreprocessModel imageData)
        {
            var list = Postprocess(imageData.imageHeight, imageData.imageWidth, ortTensor, imageData.TopPad, imageData.LeftPad);
            imageData.model.DetectionResult = Utils.GetResult(list);

        }
        private List<Detection> Postprocess(int imageHeight, int imageWidth, ReadOnlySpan<float> ortSpan, int padTop, int padLeft)
        {
            List<Rect> boxes = new List<Rect>();
            List<float> scores = new List<float>();
            List<int> class_ids = new List<int>();
            float gain = Math.Min((float)InputHeight / imageHeight, (float)InputWidth / imageWidth);
            for (int i = 0; i < _boxNums; i++)
            {
                // Move forward to confidence value of first label
                var labelOffset = i + _boxNums4;

                float bestConfidence = 0f;
                int bestLabelIndex = -1;

                // Get confidence and label for current bounding box
                for (var l = 0; l < Labels.Length; l++, labelOffset += _boxNums)
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
                float y = ortSpan[i + _boxNums] - padTop;
                float w = ortSpan[i + _boxNums2];
                float h = ortSpan[i + _boxNums3];

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
        private void PopulateNativeBuffer<T>(OrtMemoryAllocation buffer, T[] elements)
        {
            PopulateNativeBuffer(buffer.DangerousGetHandle(), elements);
        }

        private void PopulateNativeBuffer<T>(IntPtr buffer, T[] elements)
        {

            Span<T> bufferSpan;
            unsafe
            {
                bufferSpan = new Span<T>(buffer.ToPointer(), _len);
            }

            elements.AsSpan().Slice(0, _len).CopyTo(bufferSpan);
        }

        private void ClearOutput()
        {
            Span<byte> bufferSpan;
            unsafe
            {
                bufferSpan = new Span<byte>(_outputNativeAllocation.Handle.ToPointer(), (int)_outputSizeInBytes);
            }
            bufferSpan.Clear();
        }

        private void ClearInput()
        {
            Span<byte> bufferSpan;
            unsafe
            {
                bufferSpan = new Span<byte>(_inputNativeAllocation.Handle.ToPointer(), (int)_inputSizeInBytes);
            }
            bufferSpan.Clear();
        }
    }
}
