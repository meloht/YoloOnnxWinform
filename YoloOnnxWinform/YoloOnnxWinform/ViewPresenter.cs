


using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using YoloOnnxWinform.YoloWarpper;


namespace YoloOnnxWinform
{
    public class ViewPresenter
    {
        IFormProgress _formProgress;

        protected BindingList<DataModel> _bindingSource = new BindingList<DataModel>();
        protected Dictionary<string, string> _dictFile = [];
        private System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();


        public ViewPresenter(IFormProgress progress)
        {
            _formProgress = progress;
        }

        public void InitDataGridColumn(List<string> files, Dictionary<string, string> dict)
        {
            _dictFile.Clear();
            _dictFile = dict;
            _bindingSource.Clear();
            SetDataGridColumns();
            foreach (var fileName in files)
            {
                DataModel model = new DataModel();
                model.FileName = fileName;
                _bindingSource.Add(model);
            }
            if (_bindingSource.Count == 0)
            {
                MessageBox.Show("There are no pictures in this directory!", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            _formProgress.DataGridList.DataSource = _bindingSource;


        }

        protected void AddColumn(string colName, int width, DataGridView dataGridView)
        {
            DataGridViewTextBoxColumn column = new DataGridViewTextBoxColumn();
            column.HeaderText = colName;
            column.Name = $"Column{colName}";
            column.ReadOnly = true;
            column.Width = width;
            column.DataPropertyName = colName;

            dataGridView.Columns.Add(column);
        }
        protected void SetDataGridColumns()
        {
            _formProgress.DataGridList.Columns.Clear();
            AddColumn("FileName", 350, _formProgress.DataGridList);
            AddColumn("DetectionResult", 240, _formProgress.DataGridList);
            AddColumn("ExecuteTime", 200, _formProgress.DataGridList);
            AddColumn("ErrorLog", 400, _formProgress.DataGridList);

        }

        public void Process(IYoloModel yoloPredictor, ExcuteType excuteType)
        {
            if (yoloPredictor is IYoloExt && ExcuteType.Parallel == excuteType)
            {

                ProcessParallel(yoloPredictor);
            }
            else
            {
                ProcessSequence(yoloPredictor);
            }
           
        }

        private void ProcessSequence(IYoloModel yoloPredictor)
        {
            int idx = 0;
            int total = _bindingSource.Count;

            DateTime current = DateTime.Now;
            foreach (var item in _bindingSource)
            {

                try
                {
                    string filePath = _dictFile[item.FileName];
                    GetDetectResult(yoloPredictor, item, filePath);
                    var span = DateTime.Now - current;
                    if (span.TotalMilliseconds > 100)
                    {
                        current = DateTime.Now;
                        _formProgress.ShowProgress(idx * 100 / total, $"{idx}/{total}");
                    }

                }
                catch (Exception ex)
                {
                    ShowError(item, ex.Message);
                }

                idx++;
            }

            _formProgress.ShowProgress(idx * 100 / total, $"{idx}/{total}");
        }

        private void ProcessParallel(IYoloModel yoloPredictor)
        {
            IYoloExt yolo = (IYoloExt)yoloPredictor;
            yolo.PreLoadImages(_bindingSource, _dictFile);
            int idx = 0;
            int total = _bindingSource.Count;

            for (; idx < total;)
            {
                var list = yolo.GetPreLoadImages();
                foreach (var item1 in list)
                {
                    try
                    {
                        _stopwatch.Start();
                        yolo.Run(item1);

                        _stopwatch.Stop();
                        item1.model.ExecuteTime = $"{_stopwatch.Elapsed.TotalMilliseconds}ms";
                        _stopwatch.Reset();

                    }
                    catch (Exception ex)
                    {
                        ShowError(item1.model, ex.Message);
                    }
                    idx++;
                    _formProgress.ShowProgress(idx * 100 / total, $"{idx}/{total}");
                }
                Thread.Sleep(50);
            }

        }



        private void GetDetectResult(IYoloModel yoloPredictor, DataModel model, string filePath)
        {
            _stopwatch.Start();
            model.DetectionResult = yoloPredictor.DetectImage(filePath);
            _stopwatch.Stop();
            model.ExecuteTime = $"{_stopwatch.Elapsed.TotalMilliseconds}ms";
            _stopwatch.Reset();

        }

        public FileRowItem GetSelectRowData(DataGridViewRow row)
        {
            var item = row.DataBoundItem as DataModel;
            if (item != null)
            {
                return new FileRowItem() { FileName = item.FileName, FilePath = _dictFile[item.FileName] };
            }
            return null;
        }


        protected void ShowError(DataModel item, string error)
        {
            _formProgress?.DataGridList?.Invoke(new Action(() =>
            {
                if (!string.IsNullOrEmpty(item.ErrorLog))
                {
                    item.ErrorLog = $"{item.ErrorLog} {error}";
                }
                else
                {
                    item.ErrorLog = error;
                }

            }));
        }

        protected void UpdateImageItemModel(DataModel item, DataModel model)
        {
            item.ErrorLog = model.ErrorLog;
            item.DetectionResult = model.DetectionResult;
            item.ExecuteTime = model.ExecuteTime;

        }


    }
}
