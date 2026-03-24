using System;
using System.Collections.Generic;
using System.Text;

namespace YoloOnnx
{
    public struct PreResult
    {
        public float[] OutData { get; set; }

        public int TopPad { get; set; }
        public int LeftPad { get; set; }
        public float Scale { get; set; }
        public PreResult(float[] outData, int topPad, int leftPad, float scale)
        {
            OutData = outData;
            TopPad = topPad;
            LeftPad = leftPad;
            Scale = scale;
        }
    }
}
