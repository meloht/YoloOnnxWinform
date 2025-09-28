using CommImageControl;
//using Compunet.YoloSharp;
//using Compunet.YoloSharp.Plotting;
using SkiaSharp;

//using SixLabors.ImageSharp;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using YoloDotNet;
using YoloDotNet.Core;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Extensions;


namespace YoloOnnxWinform
{
    public partial class FormYoloDetect : Form, IFormProgress
    {
        private ViewPresenter _viewPresenter;

        public DataGridView DataGridList => this.dataGridView1;
        private System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();

        //private YoloPredictor _yoloPredictor;
        private DetectionDrawingOptions _drawingOptions;
        private Yolo yoloPredictor;
        public FormYoloDetect()
        {
            InitializeComponent();
            _viewPresenter = new ViewPresenter(this);
        }

        private void FormYoloDetect_Load(object sender, EventArgs e)
        {
            // _yoloPredictor = new YoloPredictor("yolo11n.onnx");

            _drawingOptions = new DetectionDrawingOptions
            {
                DrawBoundingBoxes = true,
                DrawConfidenceScore = true,
                DrawLabels = true,
                EnableFontShadow = true,

                Font = SKTypeface.Default,

                FontSize = 18,
                FontColor = SKColors.White,
                DrawLabelBackground = true,
                EnableDynamicScaling = true,
                BorderThickness = 2,

                BoundingBoxOpacity = 128,

            };
            yoloPredictor = new Yolo(new YoloOptions
            {
                OnnxModel = "yolo11n.onnx",
                ExecutionProvider = new CudaExecutionProvider(GpuId: 0, PrimeGpu: true),
                ImageResize = ImageResize.Proportional,
                SamplingOptions = new(SKFilterMode.Nearest, SKMipmapMode.None) // YoloDotNet default
            });
        }

        private void btnSelectDir_Click(object sender, EventArgs e)
        {
            if (this.folderBrowserDialog1.ShowDialog() == DialogResult.OK)
            {
                this.textboxDir.Text = this.folderBrowserDialog1.SelectedPath;
                try
                {
                    btnSelectDir.Enabled = false;


                    LoadImages(this.textboxDir.Text.Trim());
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    btnSelectDir.Enabled = true;
                }
            }
        }

        private void btnAly_Click(object sender, EventArgs e)
        {
            try
            {
                if (string.IsNullOrEmpty(this.textboxDir.Text.Trim()))
                {
                    MessageBox.Show("please select images directory first!", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    btnSelectDir.Focus();
                    return;
                }

                ProcessingButtonState(false);
                _stopwatch.Start();
                LoadImages(this.textboxDir.Text.Trim());
                ProcessImage();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void ProcessImage()
        {

            this.progressBar1.Value = 0;
            Task.Run(() =>
            {
                _viewPresenter.Process(yoloPredictor);
            });
        }

        public void ShowProgress(int val, string info)
        {
            this.Invoke(new Action(() =>
            {
                this.progressBar1.Value = val;
                this.lblProgress.Text = info;
                if (val == 100)
                {
                    ProcessingButtonState(true);
                    _stopwatch.Stop();
                    this.lblTimeCost.Text = _stopwatch.Elapsed.ToString();
                    _stopwatch.Reset();
                    MessageBox.Show("Image processing completed!");
                }
            }));
        }

        private void LoadImages(string path)
        {
            if (!Directory.Exists(path))
            {
                MessageBox.Show("the directory is not exist!", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            Dictionary<string, string> dict = new Dictionary<string, string>();
            var files = GetFilesFromDirectory(path, dict);

            if (files.Count == 0)
            {
                MessageBox.Show("the directory is not exist images!", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            _viewPresenter.InitDataGridColumn(files, dict);
        }

        private List<string> GetFilesFromDirectory(string path, Dictionary<string, string> dict)
        {
            List<string> list = new List<string>();

            GetFiles(list, path, dict);
            this.progressBar1.Value = 0;
            this.lblProgress.Text = $"0/{list.Count}";

            return list;

        }
        private void GetFiles(List<string> list, string path, Dictionary<string, string> dict)
        {
            DirectoryInfo directory = new DirectoryInfo(path);
            var files = directory.GetFiles();

            foreach (var item in files)
            {
                string filePath = item.Extension.ToLower();
                if (filePath.EndsWith(".jpg") || filePath.EndsWith(".png"))
                {
                    if (!dict.ContainsKey(item.Name))
                    {
                        list.Add(item.Name);
                        dict.Add(item.Name, item.FullName);
                    }
                }
            }
            var subDirectories = Directory.GetDirectories(path);

            foreach (string subDir in subDirectories)
            {
                GetFiles(list, subDir, dict);
            }
        }
        private void ProcessingButtonState(bool bl)
        {
            this.btnAly.Enabled = bl;
            this.btnSelectDir.Enabled = bl;
            this.showImageToolStripMenuItem.Enabled = bl;
        }

        private void showImageToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (this.dataGridView1.SelectedRows.Count == 0)
            {
                MessageBox.Show("please select a row!", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            showImageToolStripMenuItem.Enabled = false;

            try
            {
                var row = this.dataGridView1.SelectedRows[0];
                var item = _viewPresenter.GetSelectRowData(row);
                if (item != null)
                {
                    //var result = _yoloPredictor.Detect(item.FilePath);
                    //using var image = SixLabors.ImageSharp.Image.Load(item.FilePath);

                    //using var plot = result.PlotImage(image);
                    //if (plot != null)
                    //{
                    //    string folder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "temp");
                    //    if (!Directory.Exists(folder))
                    //    {
                    //        Directory.CreateDirectory(folder);
                    //    }
                    //    string path = Path.Combine(folder, item.FileName);
                    //    if (File.Exists(path))
                    //    {
                    //        File.Delete(path);
                    //    }
                    //    plot.Save(path);
                    //    FormUtils.Show(item.FileName, path);
                    //}

                    using var image = SKBitmap.Decode(item.FilePath);

                    // Run object detection inference
                    var results = yoloPredictor.RunObjectDetection(image, confidence: 0.15, iou: 0.7);

                    // Draw results
                    image.Draw(results, _drawingOptions);

                    string folder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "temp");
                    if (!Directory.Exists(folder))
                    {
                        Directory.CreateDirectory(folder);
                    }
                    string path = Path.Combine(folder, item.FileName);
                    if (File.Exists(path))
                    {
                        File.Delete(path);
                    }

                    // Save image
                    image.Save(path, SKEncodedImageFormat.Jpeg, 80);
                    FormUtils.Show(item.FileName, path);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                showImageToolStripMenuItem.Enabled = true;

            }

        }





        private void FormYoloDetect_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (this.btnAly.Enabled == false)
            {
                MessageBox.Show("the images is processing!", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }

            //if (_yoloPredictor != null)
            //{
            //    _yoloPredictor.Dispose();
            //}
        }
    }
}
