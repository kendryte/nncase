using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using NnCase.Importer;
using NnCase.IR;

namespace NnCase.Cli
{
    public class Compile
    {
        private readonly IServiceProvider _serviceProvider;

        public Compile(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public async Task Run(Options options)
        {
            var model = await File.ReadAllBytesAsync(options.Input);
            var graph = new Graph();

            switch (options.InputFormat)
            {
                case "tflite":
                    new TFLiteImporter(_serviceProvider, model, graph).Import();
                    break;
                default:
                    throw new ArgumentException($"Unsupported input format: {options.InputFormat}");
            }
        }
    }
}
