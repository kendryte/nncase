using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using RazorLight;

namespace NnCase.Converter.Converters
{
    public abstract class ScriptLayerConfig
    {

    }

    public class ScriptInputLayerConfig : ScriptLayerConfig
    {
        public string Name { get; set; }

        public long[] Dimensions { get; set; }

        public int Output { get; set; }
    }

    public class ScriptOutputLayerConfig : ScriptLayerConfig
    {
        public string Name { get; set; }

        public int Input { get; set; }
    }

    public class ScriptConv2dLayerConfig : ScriptLayerConfig
    {
        public int KernelSize { get; set; }

        public int Filters { get; set; }

        public int Stride { get; set; }

        public ActivationFunctionType Activation { get; set; }

        public int Input { get; set; }

        public int Output { get; set; }
    }

    public class ScriptSeparableConv2dLayerConfig : ScriptLayerConfig
    {
        public int KernelSize { get; set; }

        public int Filters { get; set; }

        public int Stride { get; set; }

        public ActivationFunctionType Activation { get; set; }

        public int Input { get; set; }

        public int Output { get; set; }
    }

    public class ScriptGenerationContext
    {
        public string Prefix { get; set; }

        public IReadOnlyList<ScriptLayerConfig> Layers { get; set; }

        public IReadOnlyDictionary<OutputConnector, int> Outputs { get; set; }

        public string OutputName { get; set; }
    }

    public class GraphToScriptConverter
    {
        private readonly Graph _graph;
        private readonly RazorLightEngine _templateEngine;

        public GraphToScriptConverter(Graph graph)
        {
            _graph = graph;
            _templateEngine = new RazorLightEngineBuilder()
                .UseMemoryCachingProvider()
                .UseEmbeddedResourcesProject(typeof(GraphToScriptConverter).Assembly, "Templates.Script")
                .Build();
        }

        public async Task ConvertAsync(string outputDir, string prefix)
        {
            var context = new ConvertContext();

            foreach (var layer in _graph.Outputs)
                ConvertLayer(layer, context);

            var scriptGenContext = new ScriptGenerationContext
            {
                Prefix  = prefix,
                Layers = context.Layers,
                Outputs = context.Outputs,
                OutputName = _graph.Outputs.First().Name
            };
            var code = await _templateEngine.CompileRenderAsync("Model", scriptGenContext);
            File.WriteAllText(Path.Combine(outputDir, $"{prefix}.py"), code);
        }

        private void ConvertLayer(Layer layer, ConvertContext context)
        {
            if (!context.ProcessMap.GetValueOrDefault(layer))
            {
                context.ProcessMap[layer] = true;

                foreach (var conn in layer.InputConnectors)
                {
                    var nextLayer = conn.Connection?.From.Owner;
                    if (nextLayer != null)
                        ConvertLayer(nextLayer, context);
                }

                switch (layer)
                {
                    case InputLayer l:
                        ConvertInputLayer(l, context);
                        break;
                    case OutputLayer l:
                        ConvertOutputLayer(l, context);
                        break;
                    case K210Conv2d l:
                        ConvertK210Conv2d(l, context);
                        break;
                    case K210SeparableConv2d l:
                        ConvertK210SeparableConv2d(l, context);
                        break;
                    default:
                        throw new NotSupportedException(nameof(layer));
                }
            }
        }

        private void ConvertInputLayer(InputLayer layer, ConvertContext context)
        {
            context.Layers.Add(new ScriptInputLayerConfig
            {
                Name = layer.Name,
                Dimensions = layer.Output.Dimensions.ToNHWC().ToArray(),
                Output = context.AddOutput(layer.Output)
            });
        }

        private void ConvertOutputLayer(OutputLayer layer, ConvertContext context)
        {
            context.Layers.Add(new ScriptOutputLayerConfig
            {
                Name = layer.Name,
                Input = context.Outputs[layer.Input.Connection.From]
            });
        }

        private void ConvertK210Conv2d(K210Conv2d layer, ConvertContext context)
        {
            if (layer.Conv2dType != K210Conv2dType.Conv2d)
                throw new NotSupportedException("Depthwise conv2d is not supported.");

            context.Layers.Add(new ScriptConv2dLayerConfig
            {
                KernelSize = layer.KernelWidth,
                Stride = GetStride(layer.PoolType),
                Filters = layer.OutputChannels,
                Activation = layer.FusedActivationFunction,
                Input = context.Outputs[layer.Input.Connection.From],
                Output = context.AddOutput(layer.Output)
            });
        }

        private void ConvertK210SeparableConv2d(K210SeparableConv2d layer, ConvertContext context)
        {
            context.Layers.Add(new ScriptSeparableConv2dLayerConfig
            {
                KernelSize = layer.KernelWidth,
                Stride = GetStride(layer.PoolType),
                Filters = layer.OutputChannels,
                Activation = layer.FusedActivationFunction,
                Input = context.Outputs[layer.Input.Connection.From],
                Output = context.AddOutput(layer.Output)
            });
        }

        private static int GetStride(K210PoolType poolType)
        {
            switch (poolType)
            {
                case K210PoolType.None:
                    return 1;
                case K210PoolType.LeftTop:
                    return 2;
                default:
                    throw new NotSupportedException(nameof(poolType));
            }
        }

        private class ConvertContext
        {
            public Dictionary<Layer, bool> ProcessMap = new Dictionary<Layer, bool>();

            public List<ScriptLayerConfig> Layers = new List<ScriptLayerConfig>();

            public Dictionary<OutputConnector, int> Outputs = new Dictionary<OutputConnector, int>();

            public int AddOutput(OutputConnector output)
            {
                var id = Outputs.Count;
                Outputs.Add(output, id);
                return id;
            }
        }
    }
}
