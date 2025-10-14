using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform
{
    public class Detection
    {
        public Rect Box { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; }
        public string ClassName { get; set; }
    }
}
