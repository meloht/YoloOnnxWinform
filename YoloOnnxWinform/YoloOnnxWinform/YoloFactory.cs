using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloOnnxWinform.YoloOnnx;
using YoloOnnxWinform.YoloWarpper;

namespace YoloOnnxWinform
{
    public enum YoloWarpperType
    {
        YoloSharp,
        YoloDotNet,
        YoloDetect,
        YoloDetectOrt
    }
    public static class YoloFactory
    {

        public static IYoloModel Create(YoloWarpperType yoloWarpperType)
        {
            switch (yoloWarpperType)
            {
                //case YoloWarpperType.YoloSharp:
                //    return new YoloSharpImpl();
                //case YoloWarpperType.YoloDotNet:
                //    return new YoloDotNetImpl();
                case YoloWarpperType.YoloDetect:
                    return new YoloDetectImpl();
                case YoloWarpperType.YoloDetectOrt:
                    return new YoloDetectOrtValImpl();
                default:
                    return new YoloDetectImpl();
            }
        }
    }
}

