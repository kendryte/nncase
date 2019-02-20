using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Converters;
using NnCase.Converter.Data;
using NnCase.Converter.Model;
using NnCase.Converter.Transforms;
using NnCase.Converter.Transforms.K210;

namespace NnCase.Converter
{
    public static class GraphConvert
    {
        public static async Task ExportK210Code(string modelPath, string datasetDir, string codePath)
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
                new K210GlobalAveragePoolTransform(),
                new K210FullyConnectedTransform(),
                new K210Conv2dWithMaxAvgPoolTransform(),
                new K2101x1Conv2dToFullyConnectedTransform()
            });
            var ctx = new GraphPlanContext();
            graph.Plan(ctx);
            var dim = graph.Inputs.First().Output.Dimensions.ToArray();
            var k210c = new GraphToK210Converter(graph, K210ConvertType.Code);
            await k210c.ConvertAsync(new ImageDataset(
                datasetDir,
                new[] { dim[1], dim[2], dim[3] },
                1,
                PreprocessMethods.None,
                PostprocessMethods.Normalize0To1),
                ctx,
                Path.GetDirectoryName(codePath),
                Path.GetFileNameWithoutExtension(codePath));
        }
    }
}
