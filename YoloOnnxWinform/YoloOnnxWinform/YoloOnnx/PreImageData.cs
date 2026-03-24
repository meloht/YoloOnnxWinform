using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloOnnx
{
    public struct PreImageData
    {
        public Mat LetterboxImg { get; set; }
        public int PadY { get; set; }
        public int PadX { get; set; }
        public float Scale { get; set; }

        public PreImageData(Mat letterboxImg, int padY, int padX,float scale)
        {
            LetterboxImg = letterboxImg;
            PadY = padY;
            PadX = padX;
            Scale = scale;
        }
    }

}
