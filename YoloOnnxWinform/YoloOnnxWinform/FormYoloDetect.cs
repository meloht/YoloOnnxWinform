using CommImageControl;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using YoloOnnxWinform.YoloWarpper;

namespace YoloOnnxWinform
{
    public partial class FormYoloDetect : Form, IFormProgress
    {
        private ViewPresenter _viewPresenter;

        public DataGridView DataGridList => this.dataGridView1;
        private System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();

        private IYoloModel _yoloPredictor;

        public FormYoloDetect()
        {
            InitializeComponent();
            _viewPresenter = new ViewPresenter(this);
            _yoloPredictor = YoloFactory.Create(YoloWarpperType.YoloDetectOrt);
        }

        private void FormYoloDetect_Load(object sender, EventArgs e)
        {
            _yoloPredictor.LoadModel("yolo26-white.onnx", 0.25f, 0.4f);
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
                this.lblTimeCost.Text = "00:00:00.000";
                this.progressBar1.Value = 0;

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
            Task.Run(() =>
            {
                _viewPresenter.Process(_yoloPredictor, ExcuteType.Parallel);
            });
        }

        public void ShowProgress(int val, string info)
        {
            this.Invoke(new Action(() =>
            {
                this.progressBar1.Value = val;
                this.lblProgress.Text = info;
                this.lblTimeCost.Text = _stopwatch.Elapsed.ToString(@"hh\:mm\:ss\.fff");
                if (val == 100)
                {
                    ProcessingButtonState(true);
                    _stopwatch.Stop();
                    this.lblTimeCost.Text = _stopwatch.Elapsed.ToString(@"hh\:mm\:ss\.fff");
                    _stopwatch.Reset();
                    if (this._yoloPredictor is IYoloParallel)
                    {
                        ((IYoloParallel)_yoloPredictor).EndPreload();
                    }

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
                    string path = _yoloPredictor.SaveImage(item);
                    if (!string.IsNullOrEmpty(path))
                    {
                        FormUtils.Show(item.FileName, path);
                    }
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

            if (_yoloPredictor != null)
            {
                _yoloPredictor.Dispose();
            }
        }
    }
}
