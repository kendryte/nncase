using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using CommandLine;
using NnCase.Converter.Converters;
using NnCase.Converter.Data;
using NnCase.Converter.K210.Converters;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.K210.Transforms;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Transforms;

namespace NnCase.Cli
{
    public class Options
    {
        [Option('i', "input-format", Required = true, HelpText = "Set the input format.")]
        public string InputFormat { get; set; }

        [Option('o', "output-format", Required = true, HelpText = "Set the input format.")]
        public string OutputFormat { get; set; }

        [Option("input-node", Required = false, HelpText = "Input node")]
        public string InputNode { get; set; }

        [Option("output-node", Required = false, HelpText = "Output node")]
        public string OutputNode { get; set; }

        [Option("dataset", Required = false, HelpText = "Dataset path")]
        public string Dataset { get; set; }

        [Option("dataset-format", Required = false, Default = "image", HelpText = "Dataset format")]
        public string DatasetFormat { get; set; }

        [Option("inference-type", Required = false, Default = "uint8", HelpText = "Inference type")]
        public string InferenceType { get; set; }

        [Option("postprocess", Required = false, HelpText = "Dataset postprocess")]
        public string Postprocess { get; set; }

        [Option("postprocess-op", Required = false, HelpText = "Add postprocess operator")]
        public string PostprocessOperator { get; set; }

        [Option("weights-bits", Required = false, HelpText = "Weights quantization bits", Default = 8)]
        public int WeightsBits { get; set; }

        [Option("float-fc", Required = false, Default = false, HelpText = "Use kpu based fully connected")]
        public bool FloatFc { get; set; }

        [Option("channelwise-output", Required = false, Default = false, HelpText = "Use channelwise kpu output")]
        public bool ChannelwiseOutput { get; set; }

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
                {
                    Console.WriteLine("Fatal: " + ex.Message);

                    Console.WriteLine(ex.ToString());
                }
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
                case "caffe":
                    {
                        var file = File.ReadAllBytes(options.Input);
                        var model = Caffe.NetParameter.Parser.ParseFrom(file);
                        var tfc = new CaffeToGraphConverter(model);
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
                case "tflite":
                    {
                        var file = File.ReadAllBytes(options.Input);
                        var model = tflite.Model.GetRootAsModel(new FlatBuffers.ByteBuffer(file));
                        var tfc = new TfLiteToGraphConverter(model, model.Subgraphs(0).Value);
                        tfc.Convert();
                        graph = tfc.Graph;
                        break;
                    }
                case "k210model":
                    graph = null;
                    break;
                default:
                    throw new ArgumentException("input-format");
            }

            var outputFormat = options.OutputFormat.ToLowerInvariant();
            switch (outputFormat)
            {
                case "tf":
                    {
                        var ctx = new GraphPlanContext();
                        graph.Plan(ctx);

                        using (var f = File.Open(options.Output, FileMode.Create, FileAccess.Write))
                            await ctx.SaveAsync(f);
                        break;
                    }
                case "addpad":
                    {
                        Transform.Process(graph, new Transform[] {
                            new Conv2dAddSpaceToBatchNdTransform()
                        });

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
                case "k210model":
                case "k210pb":
                    {
                        float? mean = null, std = null;
                        PostprocessMethods pm = PostprocessMethods.Normalize0To1;
                        if (options.Postprocess == "n1to1")
                            pm = PostprocessMethods.NormalizeMinus1To1;
                        else if (!string.IsNullOrWhiteSpace(options.Postprocess))
                        {
                            var match = Regex.Match(options.Postprocess, @"mean:(?<mean>(-?\d+)(\.\d+)?),std:(?<std>(-?\d+)(\.\d+)?)");
                            if (match.Success)
                            {
                                mean = float.Parse(match.Groups["mean"].Value);
                                std = float.Parse(match.Groups["std"].Value);
                            }
                            else
                            {
                                throw new ArgumentOutOfRangeException("Invalid postprocess method");
                            }
                        }

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
                        if (options.InferenceType == "float")
                        {
                            Transform.Process(graph, new Transform[] {
                                new EliminateReshapeTransform(),
                                new EliminateTwoReshapeTransform(),
                                new EliminateTensorflowReshapeTransform(),
                                new TensorflowReshapeToFlattenTransform(),
                                new GlobalAveragePoolTransform(),
                                new LeakyReluTransform(),
                                new Conv2d1x1ToFullyConnectedTransform(),
                                new ExclusiveConcatenationTransform(),
                            });
                        }
                        else
                        {
                            Transform.Process(graph, new Transform[] {
                                new EliminateReshapeTransform(),
                                new EliminateTwoReshapeTransform(),
                                new EliminateTensorflowReshapeTransform(),
                                new TensorflowReshapeToFlattenTransform(),
                                new K210SeparableConv2dTransform(),
                                new K210SpaceToBatchNdAndValidConv2dTransform(),
                                new K210SameConv2dTransform(),
                                new K210Stride2Conv2dTransform(),
                                new GlobalAveragePoolTransform(),
                                options.FloatFc ? (Transform)new DummyTransform() : new K210FullyConnectedTransform(),
                                new LeakyReluTransform(),
                                new K210Conv2dWithNonTrivialActTransform(),
                                new K210Conv2dWithMaxAvgPoolTransform(),
                                new Conv2d1x1ToFullyConnectedTransform(),
                                new K210EliminateAddRemovePaddingTransform(),
                                new QuantizedAddTransform(),
                                new QuantizedMaxPool2dTransform(),
                                new QuantizedResizeNearestNeighborTransform(),
                                new ExclusiveConcatenationTransform(),
                                new QuantizedExclusiveConcatenationTransform(),
                                new QuantizedConcatenationTransform(),
                                new EliminateQuantizeDequantizeTransform(),
                                new EliminateInputQuantizeTransform(),
                                new K210EliminateInputUploadTransform(),
                                new K210EliminateConv2dUploadTransform(),
                                new K210EliminateUploadAddPaddingTransform(),
                                new K210EliminateConv2dRequantizeTransform(),
                                options.ChannelwiseOutput ? (Transform)new K210Conv2dToChannelwiseTransform(): new DummyTransform(),
                                //new EliminateDequantizeOutputTransform()
                            });
                        }

                        {
                            var ctx = new GraphPlanContext();
                            graph.Plan(ctx);
                            if (outputFormat == "k210model")
                            {
                                var dim = graph.Inputs.First().Output.Dimensions.ToArray();

                                Dataset dataset;
                                if (options.InferenceType == "float")
                                {
                                    dataset = null;
                                }
                                else
                                {
                                    if (options.DatasetFormat == "image")
                                        dataset = new ImageDataset(
                                        options.Dataset,
                                        dim.Skip(1).ToArray(),
                                        1,
                                        PreprocessMethods.None,
                                        pm,
                                        mean,
                                        std);
                                    else if (options.DatasetFormat == "raw")
                                        dataset = new RawDataset(
                                        options.Dataset,
                                        dim.Skip(1).ToArray(),
                                        1,
                                        PreprocessMethods.None,
                                        pm,
                                        mean,
                                        std);
                                    else
                                        throw new ArgumentException("Invalid dataset format");
                                }

                                var k210c = new GraphToK210Converter(graph, options.WeightsBits);
                                await k210c.ConvertAsync(
                                    dataset,
                                    ctx,
                                    Path.GetDirectoryName(options.Output),
                                    Path.GetFileNameWithoutExtension(options.Output),
                                    options.ChannelwiseOutput);
                            }
                            else
                            {
                                using (var f = File.Open(options.Output, FileMode.Create, FileAccess.Write))
                                    await ctx.SaveAsync(f);
                            }
                        }
                        break;
                    }
                case "k210script":
                    {
                        {
                            var dim = graph.Inputs.First().Output.Dimensions.ToArray();
                            var k210c = new GraphToScriptConverter(graph);
                            await k210c.ConvertAsync(
                                Path.GetDirectoryName(options.Output),
                                Path.GetFileNameWithoutExtension(options.Output));
                        }
                        break;
                    }
                case "inference":
                    {
                        if (options.InputFormat.ToLowerInvariant() != "k210model")
                            throw new ArithmeticException("Inference mode only support k210model input.");

                        var emulator = new NnCase.Converter.K210.Emulator.K210Emulator(
                            File.ReadAllBytes(options.Input));
                        await emulator.RunAsync(options.Dataset, options.Output);
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
            var output = string.Join(',', graph.Outputs.Select(x => x.Name));

            var tmpPb = Path.GetTempFileName();
            using (var f = File.Open(tmpPb, FileMode.Create, FileAccess.Write))
                await ctx.SaveAsync(f);

            var binPath = Path.Combine(Path.GetDirectoryName(typeof(Program).Assembly.Location), "bin");
            var args = $" --input_file={tmpPb} --input_format=TENSORFLOW_GRAPHDEF --output_file={tflitePath} --output_format=TFLITE --input_shape=1,{dim[2]},{dim[3]},{dim[1]} --input_array={input} --output_arrays={output} --inference_type=FLOAT";
            using (var toco = Process.Start(new ProcessStartInfo(Path.Combine(binPath, "toco"), args)
            {
                WorkingDirectory = binPath
            }))
            {
                toco.WaitForExit();
                if (toco.ExitCode != 0)
                    throw new InvalidOperationException("Convert to tflite failed.");
            }
            File.Delete(tmpPb);
        }
    }
}
