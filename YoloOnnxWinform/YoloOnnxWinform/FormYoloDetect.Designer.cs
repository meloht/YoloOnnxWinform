namespace YoloOnnxWinform
{
    partial class FormYoloDetect
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            components = new System.ComponentModel.Container();
            panel1 = new Panel();
            lblTimeCost = new Label();
            btnAly = new Button();
            label1 = new Label();
            btnSelectDir = new Button();
            textboxDir = new TextBox();
            panel2 = new Panel();
            lblProgress = new Label();
            progressBar1 = new ProgressBar();
            groupBox1 = new GroupBox();
            dataGridView1 = new DataGridView();
            contextMenuStrip1 = new ContextMenuStrip(components);
            showImageToolStripMenuItem = new ToolStripMenuItem();
            folderBrowserDialog1 = new FolderBrowserDialog();
            panel1.SuspendLayout();
            panel2.SuspendLayout();
            groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)dataGridView1).BeginInit();
            contextMenuStrip1.SuspendLayout();
            SuspendLayout();
            // 
            // panel1
            // 
            panel1.Controls.Add(lblTimeCost);
            panel1.Controls.Add(btnAly);
            panel1.Controls.Add(label1);
            panel1.Controls.Add(btnSelectDir);
            panel1.Controls.Add(textboxDir);
            panel1.Dock = DockStyle.Top;
            panel1.Location = new Point(0, 0);
            panel1.Margin = new Padding(2, 3, 2, 3);
            panel1.Name = "panel1";
            panel1.Size = new Size(1116, 49);
            panel1.TabIndex = 0;
            // 
            // lblTimeCost
            // 
            lblTimeCost.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            lblTimeCost.AutoSize = true;
            lblTimeCost.Location = new Point(963, 17);
            lblTimeCost.Margin = new Padding(2, 0, 2, 0);
            lblTimeCost.Name = "lblTimeCost";
            lblTimeCost.Size = new Size(80, 17);
            lblTimeCost.TabIndex = 4;
            lblTimeCost.Text = "00:00:00.000";
            // 
            // btnAly
            // 
            btnAly.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btnAly.Location = new Point(766, 13);
            btnAly.Margin = new Padding(2, 3, 2, 3);
            btnAly.Name = "btnAly";
            btnAly.Size = new Size(73, 26);
            btnAly.TabIndex = 3;
            btnAly.Text = "Analyse";
            btnAly.UseVisualStyleBackColor = true;
            btnAly.Click += btnAly_Click;
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            label1.AutoSize = true;
            label1.Location = new Point(844, 17);
            label1.Margin = new Padding(2, 0, 2, 0);
            label1.Name = "label1";
            label1.Size = new Size(117, 17);
            label1.TabIndex = 2;
            label1.Text = "Execute Cost Time:";
            // 
            // btnSelectDir
            // 
            btnSelectDir.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btnSelectDir.Location = new Point(581, 13);
            btnSelectDir.Margin = new Padding(2, 3, 2, 3);
            btnSelectDir.Name = "btnSelectDir";
            btnSelectDir.Size = new Size(180, 26);
            btnSelectDir.TabIndex = 1;
            btnSelectDir.Text = "Select Images Directory";
            btnSelectDir.UseVisualStyleBackColor = true;
            btnSelectDir.Click += btnSelectDir_Click;
            // 
            // textboxDir
            // 
            textboxDir.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            textboxDir.Location = new Point(13, 14);
            textboxDir.Margin = new Padding(2, 3, 2, 3);
            textboxDir.Name = "textboxDir";
            textboxDir.ReadOnly = true;
            textboxDir.Size = new Size(562, 23);
            textboxDir.TabIndex = 0;
            // 
            // panel2
            // 
            panel2.Controls.Add(lblProgress);
            panel2.Controls.Add(progressBar1);
            panel2.Dock = DockStyle.Bottom;
            panel2.Location = new Point(0, 585);
            panel2.Margin = new Padding(2, 3, 2, 3);
            panel2.Name = "panel2";
            panel2.Size = new Size(1116, 34);
            panel2.TabIndex = 1;
            // 
            // lblProgress
            // 
            lblProgress.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            lblProgress.AutoSize = true;
            lblProgress.Location = new Point(1040, 8);
            lblProgress.Margin = new Padding(2, 0, 2, 0);
            lblProgress.Name = "lblProgress";
            lblProgress.Size = new Size(27, 17);
            lblProgress.TabIndex = 1;
            lblProgress.Text = "0/0";
            // 
            // progressBar1
            // 
            progressBar1.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            progressBar1.Location = new Point(7, 4);
            progressBar1.Margin = new Padding(2, 3, 2, 3);
            progressBar1.Name = "progressBar1";
            progressBar1.Size = new Size(1024, 25);
            progressBar1.TabIndex = 0;
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(dataGridView1);
            groupBox1.Dock = DockStyle.Fill;
            groupBox1.Location = new Point(0, 49);
            groupBox1.Margin = new Padding(2, 3, 2, 3);
            groupBox1.Name = "groupBox1";
            groupBox1.Padding = new Padding(2, 3, 2, 3);
            groupBox1.Size = new Size(1116, 536);
            groupBox1.TabIndex = 2;
            groupBox1.TabStop = false;
            groupBox1.Text = "Result";
            // 
            // dataGridView1
            // 
            dataGridView1.AllowUserToAddRows = false;
            dataGridView1.AllowUserToDeleteRows = false;
            dataGridView1.AllowUserToOrderColumns = true;
            dataGridView1.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            dataGridView1.ContextMenuStrip = contextMenuStrip1;
            dataGridView1.Dock = DockStyle.Fill;
            dataGridView1.Location = new Point(2, 19);
            dataGridView1.Margin = new Padding(2, 3, 2, 3);
            dataGridView1.MultiSelect = false;
            dataGridView1.Name = "dataGridView1";
            dataGridView1.ReadOnly = true;
            dataGridView1.RowHeadersWidth = 51;
            dataGridView1.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            dataGridView1.Size = new Size(1112, 514);
            dataGridView1.TabIndex = 0;
            // 
            // contextMenuStrip1
            // 
            contextMenuStrip1.Items.AddRange(new ToolStripItem[] { showImageToolStripMenuItem });
            contextMenuStrip1.Name = "contextMenuStrip1";
            contextMenuStrip1.Size = new Size(145, 26);
            // 
            // showImageToolStripMenuItem
            // 
            showImageToolStripMenuItem.Name = "showImageToolStripMenuItem";
            showImageToolStripMenuItem.Size = new Size(144, 22);
            showImageToolStripMenuItem.Text = "ShowImage";
            showImageToolStripMenuItem.Click += showImageToolStripMenuItem_Click;
            // 
            // FormYoloDetect
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1116, 619);
            Controls.Add(groupBox1);
            Controls.Add(panel2);
            Controls.Add(panel1);
            Margin = new Padding(2, 3, 2, 3);
            Name = "FormYoloDetect";
            Text = "FormYoloDetect";
            FormClosing += FormYoloDetect_FormClosing;
            Load += FormYoloDetect_Load;
            panel1.ResumeLayout(false);
            panel1.PerformLayout();
            panel2.ResumeLayout(false);
            panel2.PerformLayout();
            groupBox1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)dataGridView1).EndInit();
            contextMenuStrip1.ResumeLayout(false);
            ResumeLayout(false);
        }

        #endregion

        private Panel panel1;
        private Panel panel2;
        private GroupBox groupBox1;
        private DataGridView dataGridView1;
        private TextBox textboxDir;
        private Label lblProgress;
        private ProgressBar progressBar1;
        private Button btnSelectDir;
        private Label lblTimeCost;
        private Button btnAly;
        private Label label1;
        private FolderBrowserDialog folderBrowserDialog1;
        private ContextMenuStrip contextMenuStrip1;
        private ToolStripMenuItem showImageToolStripMenuItem;
    }
}