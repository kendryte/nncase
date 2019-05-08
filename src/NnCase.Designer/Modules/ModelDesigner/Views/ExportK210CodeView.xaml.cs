using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using NnCase.Converter;
using NnCase.Converter.Converters;
using NnCase.Converter.Data;
using NnCase.Converter.K210.Converters;
using NnCase.Converter.K210.Transforms;
using NnCase.Converter.Model;
using NnCase.Converter.Transforms;
using Ookii.Dialogs.Wpf;
using Transform = NnCase.Converter.Transforms.Transform;

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
                await ExportK210Code(_modelPath.Text, _datasetDir.Text, _exportPath.Text);
                MessageBox.Show("Export completed.", "NnCase", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch(Exception ex)
            {
                MessageBox.Show(ex.Message, "NnCase", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private static async Task ExportK210Code(string modelPath, string datasetDir, string codePath)
        {
            var file = File.ReadAllBytes(modelPath);
            var model = tflite.Model.GetRootAsModel(new FlatBuffers.ByteBuffer(file));
            var tfc = new TfLiteToGraphConverter(model, model.Subgraphs(0).Value);
            tfc.Convert();
            var graph = tfc.Graph;
            Transform.Process(graph, new Transform[] {
                new K210SeparableConv2dTransform(),
                new K210SpaceToBatchNdAndValidConv2dTransform(),
                new K210SameConv2dTransform(),
                new K210Stride2Conv2dTransform(),
                new GlobalAveragePoolTransform(),
                new K210FullyConnectedTransform(),
                new K210Conv2dWithMaxAvgPoolTransform(),
                new Conv2d1x1ToFullyConnectedTransform()
            });
            var ctx = new GraphPlanContext();
            graph.Plan(ctx);
            var dim = graph.Inputs.First().Output.Dimensions.ToArray();
            var k210c = new GraphToK210Converter(graph, 16);
            await k210c.ConvertAsync(new ImageDataset(
                datasetDir,
                new[] { dim[1], dim[2], dim[3] },
                1,
                PreprocessMethods.None,
                PostprocessMethods.Normalize0To1),
                ctx,
                Path.GetDirectoryName(codePath),
                Path.GetFileNameWithoutExtension(codePath),
                false);
        }
    }
}
