using CommImageControl;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CommImageControl
{
    public partial class FormShowImage : Form
    {
        private bool isImage = false;
        private string _filePath;
        public FormShowImage(string fileName, string filePath)
        {

            InitializeComponent();
            _filePath = filePath;
            this.DoubleBuffered = true;
            this.SetStyle(ControlStyles.OptimizedDoubleBuffer |
                          ControlStyles.AllPaintingInWmPaint |
                          ControlStyles.UserPaint, true);
            this.UpdateStyles();

            this.Text = fileName;
            isImage = false;

            if (filePath != null && filePath.Length > 0)
            {
                isImage = true;
                this.pictureBox1.Image = Image.FromFile(filePath);
            }
        }

        private void ApplyZoom(float zoomFactor)
        {
            if (!isImage) return;

            // 限制缩放范围
            zoomFactor = Math.Max(0.1f, Math.Min(10.0f, zoomFactor));

            // 计算新尺寸并调整PictureBox
            int newWidth = Math.Max((int)(panel1.ClientSize.Width * zoomFactor), this.panelImage.Width - 3);
            int newHeight = Math.Max((int)(panel1.ClientSize.Height * zoomFactor), this.panelImage.Height - 3);

            panel1.Size = new Size(newWidth, newHeight);


        }

        private void toolStripBtnZoomIn_Click(object sender, EventArgs e)
        {
            float zoomFactor = 1.1f;
            ApplyZoom(zoomFactor);
        }

        private void toolStripBtnZoomOut_Click(object sender, EventArgs e)
        {
            float zoomFactor = 0.9f;
            ApplyZoom(zoomFactor);
        }

        private void FormShowImage_SizeChanged(object sender, EventArgs e)
        {
            panel1.Size = new Size(this.panelImage.Size.Width - 3, this.panelImage.Height - 3);

        }



        private void FormShowImage_FormClosed(object sender, FormClosedEventArgs e)
        {
            if (this.pictureBox1.Image != null)
            {
                this.pictureBox1.Image.Dispose();
                this.pictureBox1.Image = null;
                if (!string.IsNullOrEmpty(_filePath) && File.Exists(_filePath))
                {
                    File.Delete(_filePath);
                }
            }
           
        }
    }
}
