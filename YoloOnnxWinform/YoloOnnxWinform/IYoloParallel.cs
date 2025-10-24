using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace YoloOnnxWinform
{
    public interface IYoloParallel
    {
        void PreLoadImages(BindingList<DataModel> list, Dictionary<string, string> dict);

        ImagePreprocessModel[] GetPreLoadImages();

        void Run(ImagePreprocessModel model);

        void EndPreload();
    }


}
