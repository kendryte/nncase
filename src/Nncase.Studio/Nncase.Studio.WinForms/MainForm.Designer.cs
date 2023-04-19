namespace Nncase.Studio.WinForms
{
    partial class MainForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
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
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            blazorWebView = new Microsoft.AspNetCore.Components.WebView.WindowsForms.BlazorWebView();
            SuspendLayout();
            // 
            // blazorWebView
            // 
            blazorWebView.Dock = DockStyle.Fill;
            blazorWebView.HostPage = "wwwroot\\index.html";
            blazorWebView.Location = new Point(0, 0);
            blazorWebView.Name = "blazorWebView";
            blazorWebView.Size = new Size(800, 450);
            blazorWebView.TabIndex = 0;
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(800, 450);
            Controls.Add(blazorWebView);
            Name = "MainForm";
            Text = "Nncase Studio";
            ResumeLayout(false);
        }

        #endregion

        private Microsoft.AspNetCore.Components.WebView.WindowsForms.BlazorWebView blazorWebView;
    }
}
