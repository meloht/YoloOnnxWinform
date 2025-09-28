//using Compunet.YoloSharp;
//using Compunet.YoloSharp;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloDotNet;

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
            AddColumn("DetectionResult", 200, _formProgress.DataGridList);
            AddColumn("ExecuteTime", 200, _formProgress.DataGridList);
            AddColumn("ErrorLog", 400, _formProgress.DataGridList);

        }

        public void Process(Yolo yoloPredictor)
        {
            int idx = 0;
            int total = _bindingSource.Count;
            DataModel[] items = new DataModel[10];
            DataModel[] models = new DataModel[10];
            int i = 0;
            int count = 0;
            DateTime current = DateTime.Now;
            foreach (var item in _bindingSource)
            {
                i = idx % 10;
                if (i == 0 && count > 0)
                {
                    count = 0;
                    UpdateModels(items, models);
                }
                try
                {
                    string filePath = _dictFile[item.FileName];
                    DataModel viewModel = GetDetectResult(yoloPredictor, filePath);
                    var span = DateTime.Now - current;
                    if (span.TotalMilliseconds > 300)
                    {
                        current = DateTime.Now;
                        _formProgress.ShowProgress(idx * 100 / total, $"{idx}/{total}");
                    }


                    i = idx % 10;
                    models[i] = viewModel;
                    items[i] = item;
                    count++;

                }
                catch (Exception ex)
                {
                    ShowError(item, ex.Message);
                }

                idx++;
            }
            UpdateModels(items, models);

            _formProgress.ShowProgress(idx * 100 / total, $"{idx}/{total}");
        }

        private DataModel GetDetectResult(Yolo yoloPredictor, string filePath)
        {
            DataModel model = new DataModel();
           
            _stopwatch.Start();
            using var image = SKBitmap.Decode(filePath);
            var data = yoloPredictor.RunObjectDetection(image, confidence: 0.35, iou: 0.7);
            _stopwatch.Stop();
            model.DetectionResult = data.Count.ToString();
            model.ExecuteTime = $"{_stopwatch.Elapsed.TotalMilliseconds}ms";
            _stopwatch.Reset();
            return model;
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

        protected void UpdateModels(DataModel[] item, DataModel[] model)
        {
            _formProgress?.DataGridList?.Invoke(new Action(() =>
            {
                for (int i = 0; i < item.Length; i++)
                {
                    if (item[i] == null || model[i] == null)
                    {
                        continue;
                    }
                    UpdateImageItemModel(item[i], model[i]);
                    item[i] = null;
                    model[i] = null;
                }
            }));
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
