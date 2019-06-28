using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using CommandLine;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;

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

    public class Interface : IHostedService
    {
        private readonly CommandArgsOptions _cmdArgs;
        private readonly Parser _parser;
        private readonly IServiceProvider _serviceProvider;

        public Interface(IOptions<CommandArgsOptions> options, IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
            _cmdArgs = options.Value;
            _parser = new Parser();
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            Options options = null;
            _parser.ParseArguments<Options>(_cmdArgs.Args)
                .WithParsed(o => options = o);
            if (options == null) return;

            if (options.OutputFormat == "inference")
            {
                await _serviceProvider.GetRequiredService<Inference>().Run(options);
            }
            else
            {
                await _serviceProvider.GetRequiredService<Compile>().Run(options);
            }
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            return Task.CompletedTask;
        }
    }
}
