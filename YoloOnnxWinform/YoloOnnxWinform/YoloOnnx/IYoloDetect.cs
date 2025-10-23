using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform.YoloOnnx
{
    public interface IYoloDetect: IDisposable
    {
        void DrawDetections(Mat inputImage, List<Detection> list);
        List<Detection> Run(Mat inputImage);
        void Run(ImagePreprocessModel model);
        void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict);

        ImagePreprocessModel[] GetPreLoadImages();

        void EndPreload();
    }
}
