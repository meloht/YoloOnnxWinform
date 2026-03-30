using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform
{
    public interface IYoloModel: IDisposable
    {
        void LoadModel(string modelPath, float confidence, float iou, int deviceId);
        string SaveImage(FileRowItem item);
        string DetectImage(string imgPath);
    }
}
