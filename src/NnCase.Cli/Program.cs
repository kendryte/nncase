using System;
using System.Collections.Generic;
using System.Diagnostics;
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
                case "tflite":
                    {
                        await ConvertToTFLite(graph, options.Output);
                        break;
                    }
                case "k210code":
                    {
                        if (options.InputFormat.ToLowerInvariant() != "tflite")
                        {
                            var tmpTflite = Path.GetTempFileName();
                            await ConvertToTFLite(graph, tmpTflite);

                            var file = File.ReadAllBytes(tmpTflite);
                            File.Delete(tmpTflite);
                            var model = tflite.Model.GetRootAsModel(new FlatBuffers.ByteBuffer(file));
                            var tfc = new TfLiteToGraphConverter(model, model.Subgraphs(0).Value);
                            tfc.Convert();
                            graph = tfc.Graph;
                        }

                        Transform.Process(graph, new Transform[] {
                            new K210SeprableConv2dTransform(),
                            new K210SpaceToBatchNdAndValidConv2dTransform(),
                            new K210SameConv2dTransform(),
                            new K210Stride2Conv2dTransform(),
                            new K210GlobalAveragePoolTransform(),
                            new K210FullyConnectedTransform(),
                            new K210Conv2dWithMaxPoolTransform(),
                            new K2101x1Conv2dToFullyConnectedTransform()
                        });

                        {
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
                        }
                        break;
                    }
                default:
                    throw new ArgumentException("output-format");
            }
        }

        private static async Task ConvertToTFLite(Graph graph, string tflitePath)
        {
            var ctx = new GraphPlanContext();
            graph.Plan(ctx);
            var dim = graph.Inputs.First().Output.Dimensions.ToArray();
            var input = graph.Inputs.First().Name;
            var output = graph.Outputs.First().Name;

            var tmpPb = Path.GetTempFileName();
            using (var f = File.Open(tmpPb, FileMode.Create, FileAccess.Write))
                await ctx.SaveAsync(f);
            using (var toco = Process.Start(Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), "toco"), $" --input_file={tmpPb} --input_format=TENSORFLOW_GRAPHDEF --output_file={tflitePath} --output_format=TFLITE --input_shape=1,{dim[2]},{dim[3]},{dim[1]} --input_array={input} --output_array={output} --inference_type=FLOAT"))
            {
                toco.WaitForExit();
                if (toco.ExitCode != 0)
                    throw new InvalidOperationException("Convert to tflite failed.");
            }
            File.Delete(tmpPb);
        }
    }
}
