using System;
using System.Collections.Async;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Accord.Statistics.Moving;
using NnCase.Converter.Data;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Model.Layers.K210;
using NnCase.Converter.Transforms;
using RazorLight;
using TensorFlow;

namespace NnCase.Converter.Converters
{
    public class GraphToK210Converter
    {
        private readonly Graph _graph;
        private readonly RazorLightEngine _templateEngine;

        public GraphToK210Converter(Graph graph)
        {
            _graph = graph;
            _templateEngine = new RazorLightEngineBuilder()
                .UseMemoryCachingProvider()
                .UseEmbeddedResourcesProject(typeof(GraphToK210Converter))
                .Build();
        }

        public async Task ConvertAsync(Dataset dataset, GraphPlanContext planContext)
        {
            _graph.Plan(planContext);

            var quantize = await GetMinMaxVars(dataset, planContext);
            //planContext.Reset();
            //Transform.Process(_graph, new Transform[]
            //{

            //});

            var context = new ConvertContext { Quantization = quantize };
            foreach (var layer in _graph.Outputs)
                ConvertLayer(layer, context);
        }

        private void ConvertLayer(Layer layer, ConvertContext context)
        {
            if (!context.ProcessMap.GetValueOrDefault(layer))
            {
                context.ProcessMap[layer] = true;

                switch (layer)
                {
                    case InputLayer _:
                    case OutputLayer _:
                    case AveragePool2d _:
                    case L2Normalization _:
                        break;
                    case K210Conv2d l:
                        ConvertK210Conv2d(l, context);
                        break;
                    default:
                        throw new NotSupportedException(nameof(layer));
                }

                foreach (var conn in layer.InputConnectors)
                {
                    var nextLayer = conn.Connection?.From.Owner;
                    if (nextLayer != null)
                        ConvertLayer(nextLayer, context);
                }
            }
        }

        private void ConvertK210Conv2d(K210Conv2d layer, ConvertContext context)
        {
            var config = new K210LayerConfig();
            QuantizeWeights(layer.Weights, config);
        }

        private async Task<QuantizationContext> GetMinMaxVars(Dataset dataset, GraphPlanContext planContext)
        {
            using (var session = new TFSession(planContext.TFGraph))
            {
                var connectors = new List<OutputConnector>();
                var toFetches = new List<TFOutput>();

                foreach (var output in planContext.TFOutputs)
                {
                    connectors.Add(output.Key);
                    if (!(output.Key.Owner is InputLayer))
                        toFetches.Add(output.Value);
                }

                var quantizationContext = new QuantizationContext { Outputs = connectors, PlanContext = planContext };
                await dataset.GetBatchesAsync().ForEachAsync(batch =>
                {
                    var input = batch.ToNHWC();
                    var runner = session.GetRunner();

                    runner.AddInput(planContext.Inputs.Values.First(), input);
                    foreach (var fetch in toFetches)
                        runner.Fetch(fetch);

                    var outputs = runner.Run();
                    RecordOutputs(new[] { input }.Concat(outputs).ToList(), quantizationContext);
                });

                return quantizationContext;
            }
        }

        private unsafe void RecordOutputs(IReadOnlyList<TFTensor> outputs, QuantizationContext context)
        {
            for (int i = 0; i < outputs.Count; i++)
            {
                var span = new Span<float>(outputs[i].Data.ToPointer(), (int)outputs[i].TensorByteSize / 4);
                (var min, var max) = GetMinMax(span);
                if (context.Distributions.TryGetValue(context.Outputs[i], out var range))
                    range.EMA(0.1, min, max);
                else
                    context.Distributions.Add(context.Outputs[i], new Range { Min = min, Max = max });
            }
        }

        private static void QuantizeWeights(Tensor<float> weights, K210LayerConfig config)
        {
            var buffer = weights.ToDenseTensor().Buffer.Span;
            (var min, var max) = GetMinMax(buffer);
            var scale = byte.MaxValue / (max - min);
            var bias = min * scale;

            (var mul, var shift) = ExtractValueAndShift(bias, 24);
        }

        private static (double min, double max) GetMinMax(Span<float> data)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int j = 0; j < data.Length; j++)
            {
                min = Math.Min(min, data[j]);
                max = Math.Max(max, data[j]);
            }

            return (min, max);
        }

        private static (double value, int shift) ExtractValueAndShift(double value, int maxBits)
        {
            int shift = 0;
            double mul = 0;

            if (Math.Abs(value) > 1)
            {
                mul = C.math.frexp(Math.Abs(value), ref shift);
                shift = maxBits - 1 - shift;
                mul = Math.Sign(value) * mul * (1 << (maxBits - 1));
            }

            return (mul, shift);
        }

        private class K210LayerConfig
        {
            public byte[] Weights { get; set; }

            public int ArgX { get; set; }

            public int ArgXShift { get; set; }

            public int ArgW { get; set; }

            public int ArgWShift { get; set; }
        }

        private class Range
        {
            public double Min;
            public double Max;

            public void EMA(double alpha, double min, double max)
            {
                Min = alpha * min + (1 - alpha) * Min;
                Max = alpha * max + (1 - alpha) * Max;
            }
        }

        private class QuantizationContext
        {
            public GraphPlanContext PlanContext { get; set; }

            public IReadOnlyList<OutputConnector> Outputs { get; set; }

            public Dictionary<OutputConnector, Range> Distributions { get; } = new Dictionary<OutputConnector, Range>();
        }

        private class ConvertContext
        {
            public QuantizationContext Quantization { get; set; }

            public Dictionary<Layer, bool> ProcessMap = new Dictionary<Layer, bool>();
        }
    }
}
