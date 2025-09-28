using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform
{
    public class DataModel: INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }


        private string _fileName;
        private string _detectionResult;
        private string _executeTime;
        private string _errorLog;

        public string FileName
        {
            get => _fileName;
            set
            {
                _fileName = value;
                OnPropertyChanged(nameof(FileName));
            }
        }

        public string DetectionResult
        {
            get => _detectionResult;
            set
            {
                _detectionResult = value;
                OnPropertyChanged(nameof(DetectionResult));
            }
        }

        public string ExecuteTime
        {
            get => _executeTime;
            set
            {
                _executeTime = value;
                OnPropertyChanged(nameof(ExecuteTime));
            }
        }
        public string ErrorLog
        {
            get => _errorLog;
            set
            {
                _errorLog = value;
                OnPropertyChanged(nameof(ErrorLog));
            }
        }
    }
}
