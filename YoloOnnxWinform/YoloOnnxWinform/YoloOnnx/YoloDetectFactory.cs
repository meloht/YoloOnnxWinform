using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Management;
using System.Text;
using YoloOnnxWinform;
using YoloOnnxWinform.YoloOnnx;

namespace YoloOnnx
{
    public class YoloDetectFactory
    {
        public static IYoloDetect CreateYoloDetect(string modelPath, float confidence, float iou, YoloWarpperType yoloWarpperType)
        {
            var options = BuildSessionOptions();

            var session = new InferenceSession(modelPath, options);
            var metaData = session.ModelMetadata.CustomMetadataMap;

            bool isEndToEnd = false;
            if (metaData.ContainsKey("end2end"))
            {
                isEndToEnd = bool.Parse(metaData["end2end"]);
            }

            if (isEndToEnd)
            {
                return new YoloDetectEndToEndOrtVal(session, options, confidence, iou);
            }
            else
            {
                if (yoloWarpperType == YoloWarpperType.YoloDetect)
                {
                    return new YoloDetect(session, options, confidence, iou);
                }
                else if (yoloWarpperType == YoloWarpperType.YoloDetectOrtIoBind)
                {
                    return new YoloDetectOrtIoBinding(session, options, confidence, iou);
                }
                return new YoloDetectOrtVal(session, options, confidence, iou);
            }
        }

        private static SessionOptions BuildSessionOptions()
        {
            SessionOptions session = new SessionOptions();
            session.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            session.EnableCpuMemArena=true;
            int gpuIdx = GetMainGPU();
            if (gpuIdx == -1)
            {
                return session;
            }

            session.AppendExecutionProvider_DML(gpuIdx);
            return session;

        }
        private static int GetMainGPU()
        {
            try
            {
                ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController");
                int idx = 0;
                string[] set = ["NVIDIA", "GEFORCE", "AMD", "RADEON"];
                foreach (ManagementObject mo in searcher.Get())
                {
                    string name = mo["Name"]?.ToString() ?? "";
                    if (IsContain(name, set))
                    {
                        return idx;
                    }

                    string description = mo["Description"]?.ToString() ?? "";
                    if (IsContain(description, set))
                    {
                        return idx;
                    }
                    idx++;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message);
            }
            return -1;
        }

        private static bool IsContain(string name, string[] set)
        {
            if (name != null)
            {
                foreach (var item in set)
                {
                    if (name.Contains(item))
                        return true;
                }
            }
            return false;
        }
    }
}
