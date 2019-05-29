using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Converters;
using NnCase.Converter.Data;
using NnCase.Converter.K210.Model.Hardware;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using TensorFlow;

namespace NnCase.Converter.K210.Converters
{
    public class GraphToK210Converter
    {
        private readonly Graph _graph;
        private readonly int _weightsBits;

        public GraphToK210Converter(Graph graph, int weightsBits)
        {
            if (weightsBits != 8 && weightsBits != 16)
                throw new ArgumentOutOfRangeException("weightsBits should be 8 or 16");

            _graph = graph;
            _weightsBits = weightsBits;
        }

        public async Task ConvertAsync(Dataset dataset, GraphPlanContext planContext, string outputDir, string prefix, bool channelwiseOutput)
        {
            _graph.Plan(planContext);

            var quantize = dataset == null
                ? new Stages.Quantize.QuantizationContext()
                : await Stages.Quantize.Quantizer.QuantizeAsync(dataset, planContext, channelwiseOutput);
            var convert = Stages.Convert.Converter.Convert(_graph, quantize, _weightsBits);
            var infer = Stages.Inference.InferExecutor.Infer(_graph, convert);

            Console.WriteLine($"KPU memory usage: {infer.KPUMemoryAllocator.MaxUsage * 64} B");
            Console.WriteLine($"Main memory usage: {infer.MainMemoryAllocator.MaxEnd} B");

            using (var bin = File.Open(Path.Combine(outputDir, $"{prefix}.kmodel"), FileMode.Create, FileAccess.Write))
            {
                Stages.Generate.Generator.GenerateBin(_graph, bin, _weightsBits, prefix, infer);
            }
        }
    }
}
