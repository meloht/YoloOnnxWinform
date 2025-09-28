using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform
{
    public interface IFormProgress
    {
        void ShowProgress(int val, string info);


        DataGridView DataGridList { get; }
    }
}
