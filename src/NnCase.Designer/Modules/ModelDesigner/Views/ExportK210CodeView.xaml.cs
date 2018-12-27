using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using NnCase.Converter;
using Ookii.Dialogs.Wpf;

namespace NnCase.Designer.Modules.ModelDesigner.Views
{
    /// <summary>
    /// ExportK210CodeView.xaml 的交互逻辑
    /// </summary>
    public partial class ExportK210CodeView : Window
    {
        public ExportK210CodeView()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new VistaOpenFileDialog
            {
                Title = "Select model file",
                Filter = "model (*.tflite)|*.tflite",
                CheckFileExists = true
            };

            if (dlg.ShowDialog(this) == true)
            {
                _modelPath.Text = dlg.FileName;
            }
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            var dlg = new VistaFolderBrowserDialog();

            if (dlg.ShowDialog(this) == true)
            {
                _datasetDir.Text = dlg.SelectedPath;
            }
        }

        private void Button_Click_2(object sender, RoutedEventArgs e)
        {
            var dlg = new VistaSaveFileDialog
            {
                Title = "Save code file",
                Filter = "C code (*.c)|*.c",
                ValidateNames = true
            };

            if (dlg.ShowDialog(this) == true)
            {
                _exportPath.Text = dlg.FileName;
            }
        }

        private async void Button_Click_3(object sender, RoutedEventArgs e)
        {
            try
            {
                await GraphConvert.ExportK210Code(_modelPath.Text, _datasetDir.Text, _exportPath.Text);
                MessageBox.Show("Export completed.", "NnCase", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch(Exception ex)
            {
                MessageBox.Show(ex.Message, "NnCase", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }
}
