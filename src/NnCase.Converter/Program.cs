using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using NnCase.Converter.Converters;
using NnCase.Converter.Data;
using NnCase.Converter.Model;
using NnCase.Converter.Transforms;
using NnCase.Converter.Transforms.K210;

namespace NnCase.Converter
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var file = File.ReadAllBytes(@"D:\Work\Repository\models\mobilev1_alpha_1219.tflite");
            var model = tflite.Model.GetRootAsModel(new FlatBuffers.ByteBuffer(file));
            var tfc = new TfLiteToGraphConverter(model, model.Subgraphs(0).Value);
            tfc.Convert();
            var graph = tfc.Graph;
            Transform.Process(graph, new Transform[] {
                new K210SeprableConv2dTransform(),
                new K210SpaceToBatchNdAndValidConv2dTransform(),
                new K210SameConv2dTransform(),
                new K210Stride2Conv2dTransform(),
                new K210GlobalAveragePoolTransform(),
                new K210FullyConnectedTransform(),
                new K210Conv2dWithMaxPoolTransform()
            });
            var ctx = new GraphPlanContext();
            graph.Plan(ctx);
            var k210c = new GraphToK210Converter(graph);
            await k210c.ConvertAsync(new ImageDataset(
                @"D:\Work\Repository\models\img",
                new[] { 3, 128, 128 },
                1,
                PostprocessMethods.NormalizeMinus1To1),
                ctx,
                @"D:\Work\Repository\kendryte-standalone-sdk\src\kpu6\",
                "mobilev1_alpha");

            using (var f = File.Open(@"D:\Work\Repository\models\mobilev1_alpha2.pb", FileMode.Create, FileAccess.Write))
                await ctx.SaveAsync(f);

            Console.WriteLine("Hello World!");
        }
    }
}
