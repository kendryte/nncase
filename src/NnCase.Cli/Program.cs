using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommandLine;
using NnCase.Converter.Converters;
using NnCase.Converter.Data;
using NnCase.Converter.Model;
using NnCase.Converter.Transforms;
using NnCase.Converter.Transforms.K210;

namespace NnCase.Cli
{
    public class Options
    {
        [Option('i', "input-format", Required = true, HelpText = "Set the input format.")]
        public string InputFormat { get; set; }

        [Option('o', "output-format", Required = true, HelpText = "Set the input format.")]
        public string OutputFormat { get; set; }

        [Option("dataset", Required = false, HelpText = "Dataset path")]
        public string Dataset { get; set; }

        [Value(0, MetaName = "input", HelpText = "Input path")]
        public string Input { get; set; }

        [Value(1, MetaName = "output", HelpText = "Output path")]
        public string Output { get; set; }
    }

    class Program
    {
        static async Task Main(string[] args)
        {
            AppDomain.CurrentDomain.UnhandledException += (s, e) =>
            {
                if (e.ExceptionObject is Exception ex)
                    Console.WriteLine("Fatal: " + ex.Message);
                else
                    Console.WriteLine("Fatal: Unexpected error occurred.");
                Environment.Exit(-1);
            };

            Options options = null;
            Parser.Default.ParseArguments<Options>(args)
                .WithParsed(o => options = o);
            if (options == null) return;

            Graph graph;
            switch (options.InputFormat.ToLowerInvariant())
            {
                case "tflite":
                    {
                        var file = File.ReadAllBytes(options.Input);
                        var model = tflite.Model.GetRootAsModel(new FlatBuffers.ByteBuffer(file));
                        var tfc = new TfLiteToGraphConverter(model, model.Subgraphs(0).Value);
                        tfc.Convert();
                        graph = tfc.Graph;
                        break;
                    }
                case "paddle":
                    {
                        var tfc = new PaddleToGraphConverter(options.Input);
                        tfc.Convert(0);
                        graph = tfc.Graph;
                        break;
                    }
                default:
                    throw new ArgumentException("input-format");
            }

            switch (options.OutputFormat.ToLowerInvariant())
            {
                case "tf":
                    {
                        var ctx = new GraphPlanContext();
                        graph.Plan(ctx);

                        using (var f = File.Open(options.Output, FileMode.Create, FileAccess.Write))
                            await ctx.SaveAsync(f);
                        break;
                    }
                case "k210code":
                    {
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
                        var dim = graph.Inputs.First().Output.Dimensions.ToArray();
                        var k210c = new GraphToK210Converter(graph);
                        await k210c.ConvertAsync(new ImageDataset(
                            options.Dataset,
                            new[] { dim[1], dim[2], dim[3] },
                            1,
                            PreprocessMethods.None,
                            PostprocessMethods.Normalize0To1),
                            ctx,
                            Path.GetDirectoryName(options.Output),
                            Path.GetFileNameWithoutExtension(options.Output));
                        break;
                    }
                default:
                    throw new ArgumentException("output-format");
            }
        }
    }
}
