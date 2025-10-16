using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform.YoloOnnx
{
    public class YoloDetectBase
    {
        protected Scalar[] _colorPalette;
        protected int InputWidth;
        protected int InputHeight;
        protected readonly Scalar _paddingColor;
        protected float[] rentData;

        public YoloDetectBase()
        {
            _paddingColor = new Scalar(114, 114, 114);
        }


        protected void RentDataInt(int[] dimensions)
        {
            int len = 1;
            foreach (var item in dimensions)
            {
                len = len * item;
            }
            rentData = new float[len];
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
            var topLeft = new OpenCvSharp.Point(box.X, box.Y);
            var bottomRight = new OpenCvSharp.Point(box.X + box.Width, box.Y + box.Height);

            double fontScale = 1.0;
            // 绘制边界框
            Cv2.Rectangle(img, topLeft, bottomRight, color, 2);

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
        protected (Mat letterboxImg, int topPad, int leftPad) LetterboxFor1280(Mat img)
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
            resizedImg.Dispose();
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
    }
}
