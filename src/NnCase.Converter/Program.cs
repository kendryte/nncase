using System;
using System.Collections.Generic;
using System.IO;
using NnCase.Converter.Converters;
using NnCase.Converter.Model;
using NnCase.Converter.Transforms;
using NnCase.Converter.Transforms.K210;

namespace NnCase.Converter
{
    class Program
    {
        static void Main(string[] args)
        {
            var file = File.ReadAllBytes(@"D:\Work\Repository\models\mobilev1_facenet_optimized.tflite");
            var model = tflite.Model.GetRootAsModel(new FlatBuffers.ByteBuffer(file));
            var tfc = new TfLiteToGraphConverter(model, model.Subgraphs(0).Value);
            tfc.Convert();
            var graph = tfc.Graph;
            Transform.Process(graph, new[] {
                new K210SpaceToBatchNdAndValidConv2dTransform()
            });
            var ctx = new GraphPlanContext();
            graph.Plan(ctx);

            using (var f = File.OpenWrite("test.pb"))
                ctx.Save(f);

            Console.WriteLine("Hello World!");
        }
    }
}
