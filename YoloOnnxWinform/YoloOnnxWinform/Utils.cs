using System;
using System.Collections.Generic;
using System.Text;
using YoloOnnxWinform.YoloOnnx;

namespace YoloOnnxWinform
{
    public class Utils
    {
        public static string GetResult(List<Detection> list)
        {
            if (list == null || list.Count == 0)
                return string.Empty;

            var dict = list.GroupBy(p => p.ClassName).Select(p => $"{p.Count()} {p.Key}").ToList();
            string confs = string.Join(", ", list.Select(p => Math.Round(p.Confidence, 2)));
            return $"{string.Join(", ", dict)} [{confs}]";
        }
    }
}
