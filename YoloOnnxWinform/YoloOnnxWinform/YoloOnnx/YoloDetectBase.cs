using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Management;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;


namespace YoloOnnxWinform.YoloOnnx
{
    public class YoloDetectBase
    {
        protected Scalar[] _colorPalette;
        protected int InputWidth;
        protected int InputHeight;
        protected readonly Scalar _paddingColor;
        protected float[] rentData;

        private int arrCount = 30;
        private Thread _thread;
        private List<ImagePreprocessModel> _listImg = new List<ImagePreprocessModel>();
        private volatile bool _isStart = true;
        protected BindingList<DataModel> _listName = new BindingList<DataModel>();
        protected Dictionary<string, string> _dict = new Dictionary<string, string>();
        private int _len = 0;

        int _idx = 0;
        public YoloDetectBase()
        {
            _paddingColor = new Scalar(114, 114, 114);


        }

        protected void Start()
        {
            StopLoad();
            ImageListClear();
            LoadImg(0);
            _thread = null;
            _isStart = true;
            _thread = new Thread(PreLoadImage);
            _thread.IsBackground = true;
            _thread.Start();
        }

        protected void RentDataInt(int[] dimensions)
        {
            _len = 1;
            foreach (var item in dimensions)
            {
                _len *= item;
            }
            rentData = new float[_len];
           

        }
        protected LabelModel[] MapLabelsAndColors(InferenceSession session)
        {
            var metaData = session.ModelMetadata.CustomMetadataMap;
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
        protected Scalar[] GenerateColorPalette(int count)
        {
            var rng = new Random();
            var palette = new Scalar[count];
            var colors = ColorTemplate.Get();
            for (int i = 0; i < count; i++)
            {
                palette[i] = ColorTemplate.HexToRgbaScalar(colors[i % count]);
            }
            return palette;
        }
        public void DrawDetections(Mat inputImage, List<Detection> list)
        {
            foreach (var item in list)
            {
                DrawDetections(inputImage, item.Box, item.Confidence, item.ClassId, item.ClassName);
            }
        }
        public void DrawDetections(Mat img, Rect box, float score, int classId, string className)
        {
            var color = _colorPalette[classId];

            double fontScale = 1.0;
            // 绘制边界框
            Cv2.Rectangle(img, box, color, 2);

            int height = img.Height;
            int width = img.Width;

            // 绘制标签
            string label = $"{className}: {score:F2}";
            int fontThick = 2;
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, fontScale, fontThick, out int baseline);
            var labelTop = new OpenCvSharp.Point(box.X, box.Y - 10);

            if (labelTop.Y < textSize.Height)
                labelTop.Y = box.Y + 10;

            if (labelTop.X + textSize.Width > width)
            {
                labelTop.X = labelTop.X - (labelTop.X + textSize.Width - width) - 4;
            }

            // 标签背景
            Cv2.Rectangle(img,
                new OpenCvSharp.Point(labelTop.X - 8, labelTop.Y - 8 - textSize.Height),
                new OpenCvSharp.Point(labelTop.X + textSize.Width, labelTop.Y + baseline),
                color, -1);

            // 标签文本
            Cv2.PutText(img, label, labelTop, HersheyFonts.HersheySimplex, fontScale, Scalar.White, fontThick, LineTypes.AntiAlias);
        }
        protected (Mat letterboxImg, int topPad, int leftPad) LetterboxFor1280(Mat inputImage)
        {
            // BGR转RGB
            using Mat rgbImg = new Mat();

            Cv2.CvtColor(inputImage, rgbImg, ColorConversionCodes.BGR2RGB);
            // 1. 获取原始图像尺寸
            int imgH = rgbImg.Rows;
            int imgW = rgbImg.Cols;

            // 2. 计算缩放比例（按最小比例缩放，避免图像畸变）
            float scale = Math.Min((float)InputHeight / imgH, (float)InputWidth / imgW);

            // 3. 计算缩放后的尺寸（确保按比例缩放）
            int newImgW = (int)Math.Round(imgW * scale);
            int newImgH = (int)Math.Round(imgH * scale);

            // 4. 计算填充值（左右填充、上下填充，确保最终尺寸=1280×1280）
            int padW = (InputWidth - newImgW) / 2; // 左右填充的一半
            int padH = (InputHeight - newImgH) / 2; // 上下填充的一半

            // 5. 缩放图像（若原始尺寸≠缩放后尺寸）
            using Mat resizedImg = rgbImg;
            if (imgW != newImgW || imgH != newImgH)
            {
                Cv2.Resize(rgbImg, resizedImg, new OpenCvSharp.Size(newImgW, newImgH), interpolation: InterpolationFlags.Linear);
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

            return (letterboxImg, padH, padW);
        }

        protected void OptimizedGetAllChannelData(Mat[] channels, float[] data)
        {
            if (channels == null || channels.Length == 0)
                return;

            var dataSpan = data.AsSpan();
            int index = 0;

            for (int i = 0; i < channels.Length; i++)
            {
                var channel = channels[i];
                int channelSize = channel.Rows * channel.Cols;

                var channelSpan = channel.AsSpan<float>();
                channelSpan.CopyTo(dataSpan.Slice(index, channelSize));

                index += channelSize;
            }

            foreach (var item in channels)
            {
                item.Dispose();
            }
        }
        private void GetChwArr(Mat paddedImg, float[] data)
        {
            int height = paddedImg.Height;
            int width = paddedImg.Width;
            int channels = paddedImg.Channels();
            int index = 0;
            for (int c = 0; c < channels; c++)          // 通道（R=0, G=1, B=2）
            {
                for (int h = 0; h < height; h++)  // 高度
                {
                    for (int w = 0; w < width; w++)  // 宽度
                    {
                        data[index++] = paddedImg.At<Vec3f>(h, w)[c];
                    }
                }
            }
        }
        protected unsafe void ConvertToCHW(Mat image, float[] data)
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


        protected (float[] OutData, int TopPad, int LeftPad) Preprocess(Mat inputImage)
        {

            // Letterbox处理
            (Mat paddedImg, int top, int left) = LetterboxFor1280(inputImage);

            // 归一化并转换为float数组
            paddedImg.ConvertTo(paddedImg, MatType.CV_32F, 1.0 / 255.0);

            //// 转换为CHW格式 (3, H, W)
            //var channels = paddedImg.Split();

            float[] data = rentData;
            GetChwArr(paddedImg, data);
            //OptimizedGetAllChannelData(channels, data);
            paddedImg.Dispose();
            // 添加批次维度 (1, 3, H, W)
            return (data, top, left);
        }



        protected ImagePreprocessModel PreprocessBatch(Mat inputImage, DataModel model)
        {
            // Letterbox处理
            (Mat paddedImg, int top, int left) = LetterboxFor1280(inputImage);
            using (paddedImg)
            {
                // 归一化并转换为float数组
                paddedImg.ConvertTo(paddedImg, MatType.CV_32F, 1.0 / 255.0);

                float[] data = ArrayPool<float>.Shared.Rent(_len);
                GetChwArr(paddedImg, data);

                // 添加批次维度 (1, 3, H, W)
                return new ImagePreprocessModel(inputImage.Height, inputImage.Width, model, data, top, left);
            }

        }


        private bool ImageListIsEmpty()
        {
            lock (_listImg)
            {
                return _listImg.Count == 0;
            }
        }
        private void ImageListAdd(ImagePreprocessModel model)
        {
            lock (_listImg)
            {
                _listImg.Add(model);
            }
        }
        protected ImagePreprocessModel[] GetPreImgs()
        {
            lock (_listImg)
            {
                var arr = _listImg.ToArray();
                _listImg.Clear();
                return arr;
            }
        }
        private void ImageListClear()
        {
            lock (_listImg)
            {
                _listImg.Clear();
            }
        }
        private void PreLoadImage()
        {
            _idx = 1;
            while (_isStart)
            {
                if (ImageListIsEmpty())
                {

                    for (int i = 0; _idx < _listName.Count && i < arrCount; _idx++, i++)
                    {
                        if (!_isStart)
                            break;
                        LoadImg(_idx);
                    }
                }

                Thread.Sleep(50);
            }
        }

        private void LoadImg(int idx)
        {
            string key = _listName[idx].FileName;

            string imgPath = _dict[key];

            using Mat inputImage = Cv2.ImRead(imgPath);
            var data = PreprocessBatch(inputImage, _listName[idx]);
            ImageListAdd(data);
        }

        protected void StopLoad()
        {
            _isStart = false;
            if (_thread != null && _thread.IsAlive)
            {
                _thread.Join();
            }
        }
    }
}
