using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloOnnxWinform.YoloOnnx;
using YoloOnnxWinform.YoloWarpper;
using YoloWarpper;

namespace YoloOnnxWinform
{
    public enum YoloWarpperType
    {
        YoloSharpOnnx,
        YoloDotNet,
        YoloSharp,
        YoloDetect,
        YoloDetectOrt,
        YoloDetectOrtIoBind
    }
    public static class YoloFactory
    {

        public static IYoloModel Create(YoloWarpperType yoloWarpperType)
        {
            switch (yoloWarpperType)
            {
                case YoloWarpperType.YoloSharp:
                    return new YoloSharpImpl();
                case YoloWarpperType.YoloSharpOnnx:
                    return new YoloSharpOnnxImpl();
                case YoloWarpperType.YoloDotNet:
                    return new YoloDotNetImpl();
                case YoloWarpperType.YoloDetect:
                    return new YoloDetectImpl();
                case YoloWarpperType.YoloDetectOrt:
                    return new YoloDetectOrtValImpl();
                case YoloWarpperType.YoloDetectOrtIoBind:
                    return new YoloDetectOrtIoBindingImpl();
                default:
                    return new YoloDetectOrtValImpl();
            }
        }
    }
}

