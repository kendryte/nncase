using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Data;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Model.Layers.K210;
using NnCase.Converter.Transforms;
using RazorLight;
using TensorFlow;

#if NET471
using System.Collections.Async;
#endif

namespace NnCase.Converter.Converters
{
    public class K210LayerBNConfig
    {
        public int Mul { get; set; }

        public int Shift { get; set; }

        public int Add { get; set; }
    }

    public class K210ConvLayerConfig
    {
        public ushort[] Weights { get; set; }

        public int ArgX { get; set; }

        public int ShiftX { get; set; }

        public int ArgW { get; set; }

        public int ShiftW { get; set; }

        public long ArgAdd { get; set; }

        public int KernelType { get; set; }

        public int PoolType { get; set; }

        public bool IsDepthwise { get; set; }

        public int InputChannels { get; set; }

        public int OutputChannels { get; set; }

        public K210LayerBNConfig[] BNConfigs { get; set; }

        public int InputWidth { get; set; }

        public int InputHeight { get; set; }

        public int OutputWidth { get; set; }

        public int OutputHeight { get; set; }

        public int InputGroups { get; set; }

        public int InputRowLength { get; set; }

        public int OutputGroups { get; set; }

        public int OutputRowLength { get; set; }

        public uint InputAddress { get; set; }

        public uint OutputAddress { get; set; }

        public int OutputChannelsOnTime { get; set; }

        public int LoadTimes { get; set; }

        public int OneLoadKernelsSize { get; set; }

        public int PadValue { get; set; }

        public int InputSize { get; set; }

        public int OutputSize { get; set; }

        public int ActMul { get; set; }

        public int ActShift { get; set; }
    }

    public struct K210QuantizationParam
    {
        public float Scale { get; set; }

        public float Bias { get; set; }
    }

    public enum K210LayerType
    {
        Add,
        GlobalAveragePool2d,
        Quantize,
        Dequantize,
        L2Normalization,
        Softmax,
        K210Conv = 10240,
        K210AddPadding,
        K210RemovePadding
    }

    public class K210LayerHeader
    {
        public K210LayerType Type { get; set; }

        public uint BodySize { get; set; }
    }

    public class K210Layer
    {
        public K210LayerHeader Header { get; set; }

        public object Body { get; set; }
    }

    [Flags]
    public enum K210LayerFlags
    {
        None = 0,
        MainMemoryOutput = 1
    }

    public class K210Conv2dParamAddress
    {
        public uint Layer { get; set; }

        public uint Weights { get; set; }

        public uint Bn { get; set; }

        public uint Activation { get; set; }
    }

    public class K210Conv2dLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public K210Conv2dParamAddress ParamAddress { get; set; }

        public K210ConvLayerConfig Config { get; set; }
    }

    public class AddLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAAddress { get; set; }

        public uint MainMemoryInputBAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    public class GlobalAveragePool2dLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint KernelSize { get; set; }

        public uint Channels { get; set; }
    }

    public class QuantizeLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MemoryOutputAddress { get; set; }

        public uint Width { get; set; }

        public uint Height { get; set; }

        public uint Channels { get; set; }

        public K210QuantizationParam QuantParam { get; set; }
    }

    public class DequantizeLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Count { get; set; }

        public K210QuantizationParam QuantParam { get; set; }
    }

    public class K210AddPaddingLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint KPUMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    public class K210RemovePaddingLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    public class L2NormalizationLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    public class SoftmaxLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    public class K210BinGenerationContext
    {
        public string Prefix { get; set; }

        public uint MaxStartAddress { get; set; }

        public uint MainMemoryUsage { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint MainMemoryOutputSize { get; set; }
    }

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

        public async Task ConvertAsync(Dataset dataset, GraphPlanContext planContext, string outputDir, string prefix)
        {
            _graph.Plan(planContext);

            var quantize = await GetMinMaxVars(dataset, planContext);
            var context = new ConvertContext { Quantization = quantize };
            context.ProcessMap.Clear();
            foreach (var layer in _graph.Outputs)
                FixupQuantizationRange(layer, context);

            foreach (var layer in _graph.Outputs)
                ConvertLayer(layer, context);

            context.ProcessMap.Clear();
            foreach (var layer in _graph.Outputs)
                InferenceLayer(layer, context);

            var output = context.MainMemoryMap[_graph.Outputs[0].Input.Connection.From];
            var binGenContext = new K210BinGenerationContext
            {
                Prefix = prefix,
                MaxStartAddress = context.KPUMemoryAllocator.MaxStart,
                MainMemoryUsage = context.MainMemoryAllocator.MaxEnd,
                MainMemoryOutputAddress = output.GetRawAddress(),
                MainMemoryOutputSize = output.Size
            };

            Console.WriteLine($"KPU memory usage: {context.KPUMemoryAllocator.MaxUsage * 64} B");
            Console.WriteLine($"Main memory usage: {context.MainMemoryAllocator.MaxEnd} B");

            using (var bin = File.Open(Path.Combine(outputDir, $"{prefix}.kmodel"), FileMode.Create, FileAccess.Write))
            {
                GenerateBin(bin, context.InferenceOrders, binGenContext);
            }
        }

        private void FixupQuantizationRange(Layer layer, ConvertContext context)
        {
            if (!context.ProcessMap.GetValueOrDefault(layer))
            {
                context.ProcessMap[layer] = true;

                switch (layer)
                {
                    default:
                        break;
                }

                foreach (var conn in layer.InputConnectors)
                {
                    var nextLayer = conn.Connection?.From.Owner;
                    if (nextLayer != null)
                        ConvertLayer(nextLayer, context);
                }
            }
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
                    case Concatenation _:
                        break;
                    case K210Conv2d l:
                        ConvertK210Conv2d(l, context);
                        break;
                    case Add l:
                        ConvertAdd(l, context);
                        break;
                    case GlobalAveragePool l:
                        ConvertGlobalAveragePool(l, context);
                        break;
                    case Quantize l:
                        ConvertQuantize(l, context);
                        break;
                    case Dequantize l:
                        ConvertDequantize(l, context);
                        break;
                    case K210AddPadding l:
                        ConvertK210AddPadding(l, context);
                        break;
                    case K210RemovePadding l:
                        ConvertK210RemovePadding(l, context);
                        break;
                    case L2Normalization l:
                        ConvertL2Normalization(l, context);
                        break;
                    case Softmax l:
                        ConvertSoftmax(l, context);
                        break;
                    default:
                        throw new LayerNotSupportedException(layer.GetType().Name);
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
            var config = new K210ConvLayerConfig { BNConfigs = new K210LayerBNConfig[layer.OutputChannels] };
            (var sw, var bw) = QuantizeWeights(layer.Conv2dType == K210Conv2dType.Conv2d, layer.Weights, config, _weightsBits);
            (var sx, var bx) = QuantizeInput(context.Quantization.Distributions[layer.Input.Connection.From], config);
            config.ArgAdd = (long)Math.Round(bw * bx * layer.KernelWidth * layer.KernelHeight);

            var scale = new double[layer.OutputChannels];
            for (int i = 0; i < scale.Length; i++)
                scale[i] = sw[i] * sx;

            (var so, var bo) = QuantizeBiasAndOutput(layer, layer.Bias, context.Quantization.Distributions[layer.Output], scale, config);

            config.InputChannels = layer.InputChannels;
            config.OutputChannels = layer.OutputChannels;

            config.InputWidth = layer.Input.Dimensions[3];
            config.InputHeight = layer.Input.Dimensions[2];
            (config.InputGroups, config.InputRowLength) = GetRowLayout(config.InputWidth);
            config.OutputWidth = layer.Output.Dimensions[3];
            config.OutputHeight = layer.Output.Dimensions[2];
            (config.OutputGroups, config.OutputRowLength) = GetRowLayout(config.OutputWidth);

            config.KernelType = layer.KernelWidth == 3 ? 1 : 0;
            config.IsDepthwise = layer.Conv2dType == K210Conv2dType.DepthwiseConv2d;
            config.PoolType = (int)layer.PoolType;

            config.PadValue = (int)Math.Round(-bx);

            if (layer.Conv2dType == K210Conv2dType.Conv2d)
            {
                var kernelSize = (int)layer.Weights.Length * _weightsBits / 8;
                var oneChannelSize = layer.KernelWidth * layer.KernelHeight * layer.InputChannels * _weightsBits / 8;
                var sizeLimit = _weightsBits == 8 ? 30 : 60;
                var oneLoadChannels = Math.Min(layer.OutputChannels, (int)Math.Floor(sizeLimit * 1024.0 / oneChannelSize));
                config.OneLoadKernelsSize = oneChannelSize * oneLoadChannels;
                config.LoadTimes = (int)Math.Ceiling(layer.OutputChannels / (double)oneLoadChannels);
                config.OutputChannelsOnTime = oneLoadChannels;
            }
            else
            {
                config.OneLoadKernelsSize = (int)layer.Weights.Length * _weightsBits / 8;
                config.LoadTimes = 1;
                config.OutputChannelsOnTime = layer.OutputChannels;
            }

            var inputOneLineChannels = Math.Min(layer.InputChannels, config.InputGroups);
            config.InputSize = config.InputRowLength * config.InputHeight * config.InputChannels / inputOneLineChannels;
            var outputOneLineChannels = Math.Min(layer.OutputChannels, config.OutputGroups);
            config.OutputSize = config.OutputRowLength * config.OutputHeight * config.OutputChannels / outputOneLineChannels;

            var argument = new K210Conv2dLayerArgument
            {
                Config = config,
                ParamAddress = new K210Conv2dParamAddress()
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void ConvertAdd(Add layer, ConvertContext context)
        {
            var argument = new AddLayerArgument
            {
                Channels = (uint)(layer.Output.Dimensions[1])
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void ConvertGlobalAveragePool(GlobalAveragePool layer, ConvertContext context)
        {
            var argument = new GlobalAveragePool2dLayerArgument
            {
                KernelSize = (uint)(layer.Input.Dimensions[2] * layer.Input.Dimensions[3]),
                Channels = (uint)(layer.Input.Dimensions[1])
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void ConvertQuantize(Quantize layer, ConvertContext context)
        {
            var argument = new QuantizeLayerArgument
            {
                Width = (uint)layer.Input.Dimensions[3],
                Height = (uint)layer.Input.Dimensions[2],
                Channels = (uint)layer.Input.Dimensions[1],
                QuantParam = context.Quantization.Distributions[layer.Output].GetQuantizationParam(8)
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void ConvertDequantize(Dequantize layer, ConvertContext context)
        {
            var argument = new DequantizeLayerArgument
            {
                Count = (uint)(layer.Input.Dimensions.GetSize()),
                QuantParam = context.Quantization.Distributions[layer.Input.Connection.From].GetQuantizationParam(8)
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void ConvertK210AddPadding(K210AddPadding layer, ConvertContext context)
        {
            var argument = new K210AddPaddingLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void ConvertK210RemovePadding(K210RemovePadding layer, ConvertContext context)
        {
            var argument = new K210RemovePaddingLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void ConvertL2Normalization(L2Normalization layer, ConvertContext context)
        {
            var argument = new L2NormalizationLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void ConvertSoftmax(Softmax layer, ConvertContext context)
        {
            var argument = new SoftmaxLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
            context.LayerArguments.Add(layer, argument);
        }

        private void InferenceLayer(Layer layer, ConvertContext context)
        {
            if (!context.ProcessMap.GetValueOrDefault(layer))
            {
                context.ProcessMap[layer] = true;

                foreach (var conn in layer.InputConnectors)
                {
                    var inputLayer = conn.Connection?.From.Owner;
                    if (inputLayer != null)
                        InferenceLayer(inputLayer, context);
                }

                foreach (var output in layer.OutputConnectors)
                {
                    foreach (var conn in output.Connections.Select(x => x.To.Owner))
                    {
                        switch (conn)
                        {
                            case Concatenation l:
                                AllocateInputMemoryConcatenation(l, output, context);
                                break;
                            default:
                                AllocateInputMemoryDefault(conn, output, context);
                                break;
                        }
                    }
                }

                switch (layer)
                {
                    case InputLayer _:
                    case OutputLayer _:
                    case Concatenation _:
                        break;
                    case K210Conv2d l:
                        InferenceK210Conv2d(l, context);
                        break;
                    case Add l:
                        InferenceAdd(l, context);
                        break;
                    case GlobalAveragePool l:
                        InferenceGlobalAveragePool(l, context);
                        break;
                    case Quantize l:
                        InferenceQuantize(l, context);
                        break;
                    case Dequantize l:
                        InferenceDequantize(l, context);
                        break;
                    case K210AddPadding l:
                        InferenceK210AddPadding(l, context);
                        break;
                    case K210RemovePadding l:
                        InferenceK210RemovePadding(l, context);
                        break;
                    case L2Normalization l:
                        InferenceL2Normalization(l, context);
                        break;
                    case Softmax l:
                        InferenceSoftmax(l, context);
                        break;
                    default:
                        throw new LayerNotSupportedException(layer.GetType().Name);
                }

                Console.Write($"{context.InferenceId++}: {layer.GetType().Name}");
                if (layer.InputConnectors.Count != 0)
                    Console.Write($" {string.Join("x", layer.InputConnectors[0].Dimensions.ToArray())}");
                if (layer.OutputConnectors.Count != 0)
                    Console.Write($" -> {string.Join("x", layer.OutputConnectors[0].Dimensions.ToArray())}");
                Console.WriteLine();

                foreach (var conn in layer.InputConnectors)
                {
                    var output = conn.Connection?.From;
                    if (output != null)
                    {
                        if (context.KPUMemoryMap.TryGetValue(output, out var alloc))
                            alloc.Node.Release();
                        if (context.MainMemoryMap.TryGetValue(output, out var alloc2))
                            alloc2.Node.Release();
                    }
                }
            }
        }

        private void AllocateInputMemoryConcatenation(Concatenation layer, OutputConnector input, ConvertContext context)
        {
            var totalAlloc = GetOrAllocateMainMemory(context, layer.Output);
            uint offset = 0;

            foreach (var node in layer.Inputs.Select(x => x.Connection.From))
            {
                if (context.MainMemoryMap.ContainsKey(node))
                    return;
                uint size = (uint)node.Dimensions.GetSize() * 4;
                context.MainMemoryMap.Add(node, new MemoryAllocation(totalAlloc.Node, offset, size));
                offset += size;
            }
        }

        private void AllocateInputMemoryDefault(Layer layer, OutputConnector input, ConvertContext context)
        {
            switch (layer)
            {
                case K210Conv2d _:
                    GetOrAllocateKPUMemory(context, input);
                    break;
                default:
                    GetOrAllocateMainMemory(context, input);
                    break;
            }
        }

        private void InferenceK210Conv2d(K210Conv2d layer, ConvertContext context)
        {
            var inputAlloc = context.KPUMemoryMap[layer.Input.Connection.From];
            MemoryAllocation outputAlloc;

            var argument = (K210Conv2dLayerArgument)context.LayerArguments[layer];
            argument.Config.InputAddress = inputAlloc.GetAddress();

            if (context.MainMemoryMap.TryGetValue(layer.Output, out var mainAlloc))
            {
                argument.Flags = K210LayerFlags.MainMemoryOutput;
                argument.MainMemoryOutputAddress = mainAlloc.GetAddress();
                outputAlloc = GetOrAllocateKPUMemory(context, layer.Output);
            }
            else
            {
                argument.Flags = K210LayerFlags.None;
                outputAlloc = context.KPUMemoryMap[layer.Output];
            }

            argument.Config.OutputAddress = outputAlloc.GetAddress();
            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.K210Conv },
                Body = argument
            });
        }

        private void InferenceAdd(Add layer, ConvertContext context)
        {
            var inputAAlloc = context.MainMemoryMap[layer.InputA.Connection.From];
            var inputBAlloc = context.MainMemoryMap[layer.InputB.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            var argument = (AddLayerArgument)context.LayerArguments[layer];
            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAAddress = inputAAlloc.GetAddress();
            argument.MainMemoryInputBAddress = inputBAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();

            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.Add },
                Body = argument
            });
        }

        private void InferenceGlobalAveragePool(GlobalAveragePool layer, ConvertContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            var argument = (GlobalAveragePool2dLayerArgument)context.LayerArguments[layer];
            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();

            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.GlobalAveragePool2d },
                Body = argument
            });
        }

        private void InferenceQuantize(Quantize layer, ConvertContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];

            var argument = (QuantizeLayerArgument)context.LayerArguments[layer];
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();

            if (context.MainMemoryMap.TryGetValue(layer.Output, out var mainAlloc))
            {
                argument.Flags = K210LayerFlags.MainMemoryOutput;
                argument.MemoryOutputAddress = mainAlloc.GetAddress();
            }
            else if (context.KPUMemoryMap.TryGetValue(layer.Output, out var kpuAlloc))
            {
                argument.Flags = K210LayerFlags.None;
                argument.MemoryOutputAddress = kpuAlloc.GetAddress();
            }
            else
            {
                throw new InvalidOperationException("No allocation found");
            }

            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.Quantize },
                Body = argument
            });
        }

        private void InferenceDequantize(Dequantize layer, ConvertContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            var argument = (DequantizeLayerArgument)context.LayerArguments[layer];
            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();

            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.Dequantize },
                Body = argument
            });
        }

        private void InferenceK210AddPadding(K210AddPadding layer, ConvertContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.KPUMemoryMap[layer.Output];

            var argument = (K210AddPaddingLayerArgument)context.LayerArguments[layer];
            argument.Flags = K210LayerFlags.None;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.KPUMemoryOutputAddress = outputAlloc.GetAddress();

            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.K210AddPadding },
                Body = argument
            });
        }

        private void InferenceK210RemovePadding(K210RemovePadding layer, ConvertContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            var argument = (K210RemovePaddingLayerArgument)context.LayerArguments[layer];
            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();

            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.K210RemovePadding },
                Body = argument
            });
        }

        private void InferenceL2Normalization(L2Normalization layer, ConvertContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            var argument = (L2NormalizationLayerArgument)context.LayerArguments[layer];
            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();

            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.L2Normalization },
                Body = argument
            });
        }

        private void InferenceSoftmax(Softmax layer, ConvertContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            var argument = (L2NormalizationLayerArgument)context.LayerArguments[layer];
            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();

            context.InferenceOrders.Add(new K210Layer
            {
                Header = new K210LayerHeader { Type = K210LayerType.Softmax },
                Body = argument
            });
        }

        private void GenerateBin(Stream bin, IReadOnlyList<K210Layer> layers, K210BinGenerationContext context)
        {
            var bw = new BinaryWriter(bin);

            uint version = 3;
            uint flags = _weightsBits == 8 ? 1u : 0u;
            bw.Write(version);
            bw.Write(flags);
            bw.Write(layers.Count);
            bw.Write(context.MaxStartAddress);
            bw.Write(context.MainMemoryUsage);
            bw.Write(context.MainMemoryOutputAddress);
            bw.Write(context.MainMemoryOutputSize);

            // Headers
            var fixPosition = bw.BaseStream.Position;
            bw.BaseStream.Position += 4 * 2 * layers.Count;

            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];
                // BodySize
                var beginPosition = bw.BaseStream.Position;
                GenerateBinLayerBody(bw, layer);
                layer.Header.BodySize = (uint)(bw.BaseStream.Position - beginPosition);
            }

            var newPosition = bw.BaseStream.Position;
            bw.BaseStream.Position = fixPosition;
            for (int i = 0; i < layers.Count; i++)
            {
                var header = layers[i].Header;
                bw.Write((uint)header.Type);
                bw.Write((uint)header.BodySize);
            }

            bw.BaseStream.Position = newPosition;
        }

        private void GenerateBinLayerBody(BinaryWriter bw, K210Layer layer)
        {
            switch (layer.Body)
            {
                case K210Conv2dLayerArgument l:
                    GenerateBinConv2d(bw, l);
                    break;
                case AddLayerArgument l:
                    GenerateBinAdd(bw, l);
                    break;
                case GlobalAveragePool2dLayerArgument l:
                    GenerateBinGlobalAveragePool2d(bw, l);
                    break;
                case QuantizeLayerArgument l:
                    GenerateBinQuantize(bw, l);
                    break;
                case DequantizeLayerArgument l:
                    GenerateBinDequantize(bw, l);
                    break;
                case K210AddPaddingLayerArgument l:
                    GenerateBinAddPadding(bw, l);
                    break;
                case K210RemovePaddingLayerArgument l:
                    GenerateBinRemovePadding(bw, l);
                    break;
                case L2NormalizationLayerArgument l:
                    GenerateBinL2Normalization(bw, l);
                    break;
                case SoftmaxLayerArgument l:
                    GenerateBinSoftmax(bw, l);
                    break;
                default:
                    throw new NotSupportedException(layer.Body.GetType().Name);
            }

            AlignStreamPosition(bw.BaseStream, 8);
        }

        private void GenerateBinConv2d(BinaryWriter bw, K210Conv2dLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryOutputAddress);
            // BodySize
            var fixPosition = bw.BaseStream.Position;
            bw.BaseStream.Position += 4 * 4;

            GenerateBinLayer(bw, layer.Config, layer.ParamAddress);
            GenerateBinWeights(bw, layer.Config, layer.ParamAddress);
            GenerateBinBn(bw, layer.Config, layer.ParamAddress);
            GenerateBinActivation(bw, layer.Config, layer.ParamAddress);

            var newPosition = bw.BaseStream.Position;
            bw.BaseStream.Position = fixPosition;
            bw.Write(layer.ParamAddress.Layer);
            bw.Write(layer.ParamAddress.Weights);
            bw.Write(layer.ParamAddress.Bn);
            bw.Write(layer.ParamAddress.Activation);
            bw.BaseStream.Position = newPosition;
        }

        private void GenerateBinAdd(BinaryWriter bw, AddLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryInputAAddress);
            bw.Write(layer.MainMemoryInputBAddress);
            bw.Write(layer.MainMemoryOutputAddress);
            bw.Write(layer.Channels);
        }

        private void GenerateBinGlobalAveragePool2d(BinaryWriter bw, GlobalAveragePool2dLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryInputAddress);
            bw.Write(layer.MainMemoryOutputAddress);
            bw.Write(layer.KernelSize);
            bw.Write(layer.Channels);
        }

        private void GenerateBinQuantize(BinaryWriter bw, QuantizeLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryInputAddress);
            bw.Write(layer.MemoryOutputAddress);
            bw.Write(layer.Width);
            bw.Write(layer.Height);
            bw.Write(layer.Channels);
            bw.Write(layer.QuantParam.Scale);
            bw.Write(layer.QuantParam.Bias);
        }

        private void GenerateBinDequantize(BinaryWriter bw, DequantizeLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryInputAddress);
            bw.Write(layer.MainMemoryOutputAddress);
            bw.Write(layer.Count);
            bw.Write(layer.QuantParam.Scale);
            bw.Write(layer.QuantParam.Bias);
        }

        private void GenerateBinAddPadding(BinaryWriter bw, K210AddPaddingLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryInputAddress);
            bw.Write(layer.KPUMemoryOutputAddress);
            bw.Write(layer.Channels);
        }

        private void GenerateBinRemovePadding(BinaryWriter bw, K210RemovePaddingLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryInputAddress);
            bw.Write(layer.MainMemoryOutputAddress);
            bw.Write(layer.Channels);
        }

        private void GenerateBinL2Normalization(BinaryWriter bw, L2NormalizationLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryInputAddress);
            bw.Write(layer.MainMemoryOutputAddress);
            bw.Write(layer.Channels);
        }

        private void GenerateBinSoftmax(BinaryWriter bw, SoftmaxLayerArgument layer)
        {
            bw.Write((uint)layer.Flags);
            bw.Write(layer.MainMemoryInputAddress);
            bw.Write(layer.MainMemoryOutputAddress);
            bw.Write(layer.Channels);
        }

        private void GenerateBinWeights(BinaryWriter bw, K210ConvLayerConfig layer, K210Conv2dParamAddress paramAddress)
        {
            paramAddress.Weights = AlignStreamPosition(bw.BaseStream, 128);

            if (_weightsBits == 8)
            {
                foreach (var v in layer.Weights)
                    bw.Write((byte)v);
            }
            else
            {
                foreach (var v in layer.Weights)
                    bw.Write(v);
            }
        }

        private void GenerateBinBn(BinaryWriter bw, K210ConvLayerConfig layer, K210Conv2dParamAddress paramAddress)
        {
            paramAddress.Bn = AlignStreamPosition(bw.BaseStream, 128);

            for (int j = 0; j < layer.BNConfigs.Length; j++)
            {
                var bn = layer.BNConfigs[j];
                var reg = new K210.kpu_batchnorm_argument_t();
                reg.norm_add = (uint)bn.Add;
                reg.norm_mul = (uint)bn.Mul;
                reg.norm_shift = (byte)bn.Shift;

                bw.Write(reg.Value);
            }
        }

        private void GenerateBinActivation(BinaryWriter bw, K210ConvLayerConfig layer, K210Conv2dParamAddress paramAddress)
        {
            paramAddress.Activation = AlignStreamPosition(bw.BaseStream, 256);

            var reg = new K210.kpu_activate_table_t();
            var starts = new ulong[]
            {
                    0x800000000, 0xf7d4cf4b8, 0xf8ed5a20c, 0xfa05e4f60,
                    0xfb2e05baa, 0xfc46908fe, 0xfd5f1b652, 0xfe77a63a6,
                    0xff9fc6ff0, 0xfffd4a9b7, 0, 0x7FFFFFFF0,
                    0x7FFFFFFF1, 0x7FFFFFFF2, 0x7FFFFFFF3, 0x7FFFFFFF4
            };

            for (int j = 0; j < starts.Length; j++)
            {
                ref var param = ref reg.activate_para[j];
                param.x_start = starts[j];
                param.y_mul = 0;
                param.shift_number = 0;
            }

            {
                ref var param = ref reg.activate_para[10];
                param.y_mul = (ushort)layer.ActMul;
                param.shift_number = (byte)layer.ActShift;
            }

            for (int j = 0; j < starts.Length; j++)
                bw.Write(reg.activate_para[j].Value);
            bw.Write(reg.activate_para_bias0.Value);
            bw.Write(reg.activate_para_bias1.Value);
        }

        private void GenerateBinLayer(BinaryWriter bw, K210ConvLayerConfig layer, K210Conv2dParamAddress paramAddress)
        {
            paramAddress.Layer = AlignStreamPosition(bw.BaseStream, 8);

            var reg = new K210.kpu_layer_argument_t();

            reg.interrupt_enabe = new K210.interrupt_enabe_t
            {
                depth_wise_layer = (byte)(layer.IsDepthwise ? 1 : 0)
            };
            reg.image_addr = new K210.image_addr_t
            {
                image_src_addr = (ushort)layer.InputAddress,
                image_dst_addr = (ushort)layer.OutputAddress
            };
            reg.image_channel_num = new K210.image_channel_num_t
            {
                i_ch_num = (ushort)(layer.InputChannels - 1),
                o_ch_num = (ushort)(layer.OutputChannels - 1),
                o_ch_num_coef = (ushort)(layer.OutputChannelsOnTime - 1)
            };
            reg.image_size = new K210.image_size_t
            {
                i_row_wid = (ushort)(layer.InputWidth - 1),
                i_col_high = (ushort)(layer.InputHeight - 1),
                o_row_wid = (ushort)(layer.OutputWidth - 1),
                o_col_high = (ushort)(layer.OutputHeight - 1)
            };
            reg.kernel_pool_type_cfg = new K210.kernel_pool_type_cfg_t
            {
                load_para = 1,
                kernel_type = (byte)layer.KernelType,
                pool_type = (byte)layer.PoolType,
                dma_burst_size = 15,
                pad_value = (byte)layer.PadValue
            };
            reg.kernel_load_cfg = new K210.kernel_load_cfg_t
            {
                load_coor = 1,
                load_time = (byte)(layer.LoadTimes - 1),
                para_size = (uint)layer.OneLoadKernelsSize
            };
            reg.kernel_calc_type_cfg = new K210.kernel_calc_type_cfg_t
            {
                channel_switch_addr = (ushort)(layer.InputRowLength * layer.InputHeight),
                row_switch_addr = (byte)layer.InputRowLength,
                coef_group = (byte)layer.InputGroups,
                load_act = 1
            };
            reg.write_back_cfg = new K210.write_back_cfg_t
            {
                wb_channel_switch_addr = (ushort)(layer.OutputRowLength * layer.OutputHeight),
                wb_row_switch_addr = (byte)layer.OutputRowLength,
                wb_group = (byte)layer.OutputGroups
            };
            reg.conv_value = new K210.conv_value_t
            {
                shr_w = (byte)layer.ShiftW,
                shr_x = (byte)layer.ShiftX,
                arg_w = (uint)layer.ArgW,
                arg_x = (uint)layer.ArgX
            };
            reg.conv_value2 = new K210.conv_value2_t
            {
                arg_add = (ulong)layer.ArgAdd
            };
            reg.dma_parameter = new K210.dma_parameter_t
            {
                channel_byte_num = (ushort)(layer.OutputWidth * layer.OutputHeight - 1),
                dma_total_byte = (uint)(layer.OutputWidth * layer.OutputHeight * layer.OutputChannels - 1)
            };

            bw.Write(reg.interrupt_enabe.Value);
            bw.Write(reg.image_addr.Value);
            bw.Write(reg.image_channel_num.Value);
            bw.Write(reg.image_size.Value);
            bw.Write(reg.kernel_pool_type_cfg.Value);
            bw.Write(reg.kernel_load_cfg.Value);
            bw.Write(reg.kernel_offset.Value);
            bw.Write(reg.kernel_calc_type_cfg.Value);
            bw.Write(reg.write_back_cfg.Value);
            bw.Write(reg.conv_value.Value);
            bw.Write(reg.conv_value2.Value);
            bw.Write(reg.dma_parameter.Value);
        }

        private uint AlignStreamPosition(Stream stream, int alignment)
        {
            var cnt = stream.Position;
            var rem = cnt % alignment;
            if (rem != 0)
                stream.Position = cnt + (alignment - rem);
            return (uint)stream.Position;
        }

        private static (int groups, int rowLength) GetRowLayout(int width)
        {
            int groups, rowLength;

            if (width <= 16)
            {
                groups = 4;
                rowLength = 1;
            }
            else if (width <= 32)
            {
                groups = 2;
                rowLength = 1;
            }
            else
            {
                groups = 1;
                rowLength = (int)Math.Ceiling(width / 64.0);
            }

            return (groups, rowLength);
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

#if NET471
                await dataset.GetBatchesAsync().ForEachAsync(async batch =>
#else
                await foreach (var batch in dataset.GetBatchesAsync())
#endif
                {
                    var input = batch.ToNHWC();
                    var runner = session.GetRunner();

                    runner.AddInput(planContext.Inputs.Values.First(), input);
                    foreach (var fetch in toFetches)
                        runner.Fetch(fetch);

                    var outputs = runner.Run();
                    RecordOutputs(new[] { input }.Concat(outputs).ToList(), quantizationContext);
                }
#if NET471
                );
#endif
                return quantizationContext;
            }
        }

        private unsafe void RecordOutputs(IReadOnlyList<TFTensor> outputs, QuantizationContext context)
        {
            for (int i = 0; i < outputs.Count; i++)
            {
                var conn = context.Outputs[i];
                var span = new Span<float>(outputs[i].Data.ToPointer(), (int)outputs[i].TensorByteSize / 4);
                var newRange = GetRange(span);
                if (context.Distributions.TryGetValue(conn, out var range))
                    context.Distributions[conn] = range.EMA(0.01, newRange);
                else
                    context.Distributions.Add(conn, newRange);
            }
        }

        private static (double[] scale, double bias) QuantizeWeights(bool isConv2d, Tensor<float> weights, K210ConvLayerConfig config, int weightsBits)
        {
#if CHANNEL_WISE
            var kernels = weights.ToDenseTensor().Buffer.Span;
            var channels = weights.Dimensions[isConv2d ? 0 : 1];
            var channelSize = weights.Dimensions.GetSize() / channels;

            var totalRange = GetRange(kernels);
            var scales = new double[channels];

            for (int i = 0; i < channels; i++)
            {
                double s;
                var buffer = kernels.Slice(i * channelSize, channelSize);
                var range = GetRange(buffer);

                var s1 = totalRange.Max / range.Max;
                var s2 = totalRange.Min / range.Min;
                s = (s1 < 0 || s2 < 0) ? Math.Max(s1, s2) : Math.Min(s1, s2);

                Debug.Assert(s > 0);
                for (int j = 0; j < buffer.Length; j++)
                    buffer[j] = (float)(buffer[j] * s);
                scales[i] = s;
            }

            (var scale, var bias) = GetRange(kernels).GetScaleBias(weightsBits);

            (var mul, var shift) = ExtractValueAndShift(bias, 24, 15);
            config.Weights = Quantize(kernels, scale, bias, weightsBits);
            config.ArgX = (int)Math.Round(mul);
            config.ShiftX = shift;

            for (int i = 0; i < scales.Length; i++)
                scales[i] *= scale;
            return (scales, bias);
#else
            var buffer = weights.ToDenseTensor().Buffer.Span;
            (var scale, var bias) = GetRange(buffer).GetScaleBias();

            (var mul, var shift) = ExtractValueAndShift(bias, 24, 15);
            config.Weights = Quantize(buffer, scale, bias);
            config.ArgX = (int)Math.Round(mul);
            config.ShiftX = shift;
            return (Enumerable.Repeat(scale, weights.Dimensions[0]).ToArray(), bias);
#endif
        }

        private static (double scale, double bias) QuantizeInput(Range range, K210ConvLayerConfig config)
        {
            (var scale, var bias) = range.GetScaleBias(8);
            (var mul, var shift) = ExtractValueAndShift(bias, 24, 15);
            config.ArgW = (int)Math.Round(mul);
            config.ShiftW = shift;
            return (scale, bias);
        }

        private static (double scale, double bias) QuantizeBiasAndOutput(K210Conv2d layer, Tensor<float> bias, Range range, double[] scale, K210ConvLayerConfig config)
        {
            (var so, var bo) = range.GetScaleBias(8);
#if CHANNEL_WISE
            var upshift = 10;
            var postMul = Math.Pow(2, upshift);

            for (int i = 0; i < config.BNConfigs.Length; i++)
            {
                var b = bias[i];

                var scomb = so * postMul / scale[i];

                (var mul, var shift) = ExtractValueAndShift(scomb, 22, 15);

                config.BNConfigs[i] = new K210LayerBNConfig
                {
                    Mul = (int)Math.Round(mul),
                    Shift = shift,
                    Add = (int)Math.Round((b * so - bo) * postMul)
                };
            }
#else
            var scomb = so / scale[0];

            (var mul, var shift) = ExtractValueAndShift(scomb, 22, 255);
            var upscale = shift - 15;
            Debug.Assert(upscale >= 0);
            var postMul = Math.Round(mul) / mul * Math.Pow(2, upscale);

            for (int i = 0; i < config.BNConfigs.Length; i++)
            {
                var b = bias[i];

                config.BNConfigs[i] = new K210LayerBNConfig
                {
                    Mul = (int)Math.Round(mul),
                    Shift = 15,
                    Add = (int)Math.Round((b * so - bo) * postMul)
                };
            }
#endif
            QuantizeActivation(layer, postMul, range, config);
            return (so, bo);
        }

        private static void QuantizeActivation(K210Conv2d layer, double postMul, Range range, K210ConvLayerConfig config)
        {
            Func<double, double> invAct;
            switch (layer.FusedActivationFunction)
            {
                case ActivationFunctionType.Linear:
                case ActivationFunctionType.Relu:
                case ActivationFunctionType.Relu6:
                    invAct = x => x;
                    break;
                default:
                    throw new NotSupportedException($"Activation of {layer.FusedActivationFunction} is not supported.");
            }

            var yTable = Generator.Step(range.Min, range.Max, 15);
            var xTable = yTable.Select(invAct).Select(x => x * postMul).ToArray();

            (var mul, var shift) = ExtractValueAndShift(1 / postMul, 16, 20);
            config.ActMul = (int)Math.Round(mul);
            config.ActShift = shift;
        }

        private static double Quantize(ReadOnlySpan<float> data, Span<ushort> dest, double scale, double bias, int weightsBits)
        {
            ushort max = (ushort)((1 << weightsBits) - 1);

            for (int i = 0; i < data.Length; i++)
                dest[i] = (ushort)
#if NET471
                    FxExtensions
#else
                    Math
#endif
                    .Clamp(Math.Round(data[i] * scale - bias), 0, max);

            var diff = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
                diff[i] = Math.Abs(((dest[i] + bias) / scale) - data[i]);
            var avg = diff.Max();
            return avg;
        }

        private static ushort[] Quantize(ReadOnlySpan<float> data, double scale, double bias, int weightsBits)
        {
            var q = new ushort[data.Length];
            Quantize(data, q, scale, bias, weightsBits);
            return q;
        }

        private static Range GetRange(Span<float> data)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int j = 0; j < data.Length; j++)
            {
                if (Math.Abs(data[j]) > 100) continue;
                min = Math.Min(min, data[j]);
                max = Math.Max(max, data[j]);
            }

            //if (Math.Abs(min) > 100 || max > 100)
            //    return new Range { Min = -1, Max = 1 };

            return new Range { Min = min, Max = max };
        }

        private static (double value, int shift) ExtractValueAndShift(double value, int maxBits, int maxShift)
        {
            int shift = 0;
            double mul = 0;

            if (Math.Abs(value) > 1)
            {
                var mulShift = 0;
                mul = C.math.frexp(value, ref mulShift);
                shift = Math.Min(maxShift, maxBits - 1 - mulShift);
                mul = mul * Math.Pow(2, shift + mulShift);
            }
            else if (value == 0)
            {
                mul = shift = 0;
            }
            else
            {
                var mulShift = 0;
                mul = C.math.frexp(value, ref mulShift);
                shift = Math.Min(maxShift + mulShift, maxBits - 1);
                mul = mul * Math.Pow(2, shift);
                shift -= mulShift;
            }

            Debug.Assert(Math.Abs(mul) < Math.Pow(2, maxBits - 1));
            Debug.Assert(shift <= maxShift);
            Debug.Assert(Math.Abs(value - mul * Math.Pow(2, -shift)) <= double.Epsilon);
            return (mul, shift);
        }

        private struct Range
        {
            public double Min;
            public double Max;

            public Range EMA(double alpha, Range range)
            {
                return new Range { Min = alpha * range.Min + (1 - alpha) * Min, Max = alpha * range.Max + (1 - alpha) * Max };
            }

            public (double scale, double bias) GetScaleBias(int maxBits)
            {
                var scale = ((1 << maxBits) - 1) / (Max - Min);
                var bias = Math.Round(Min * scale);
                return (scale, bias);
            }

            public override string ToString()
            {
                return $"{Min}, {Max}";
            }

            public K210QuantizationParam GetQuantizationParam(int maxBits)
            {
                (var scale, var bias) = GetScaleBias(maxBits);
                return new K210QuantizationParam
                {
                    Scale = (float)(1 / scale),
                    Bias = (float)(bias / scale)
                };
            }

            public void UnionWith(Range range)
            {
                Min = Math.Min(Min, range.Min);
                Max = Math.Max(Max, range.Max);
            }
        }

        private MemoryAllocation GetOrAllocateKPUMemory(ConvertContext context, OutputConnector output)
        {
            if (!context.KPUMemoryMap.TryGetValue(output, out var alloc))
            {
                var dimensions = output.Dimensions;
                (var groups, var rowLength) = GetRowLayout(dimensions[3]);
                var oneLineChannels = Math.Min(dimensions[1], groups);
                var size = rowLength * dimensions[2] * dimensions[1] / oneLineChannels;
                alloc = new MemoryAllocation(context.KPUMemoryAllocator.Allocate((uint)size));
                context.KPUMemoryMap.Add(output, alloc);
            }
            else
            {
                alloc.Node.AddRef();
            }

            return alloc;
        }

        private MemoryAllocation GetOrAllocateMainMemory(ConvertContext context, OutputConnector output)
        {
            if (!context.MainMemoryMap.TryGetValue(output, out var alloc))
            {
                uint elementSize;
                switch (output.Owner)
                {
                    case K210Conv2d _:
                    case K210AddPadding _:
                    case K210RemovePadding _:
                    case Quantize _:
                        elementSize = 1;
                        break;
                    default:
                        elementSize = 4;
                        break;
                }

                var dimensions = output.Dimensions;
                alloc = new MemoryAllocation(context.MainMemoryAllocator.Allocate((uint)dimensions.GetSize() * elementSize));
                context.MainMemoryMap.Add(output, alloc);
            }
            else
            {
                alloc.Node.AddRef();
            }

            return alloc;
        }

        private class QuantizationContext
        {
            public GraphPlanContext PlanContext { get; set; }

            public IReadOnlyList<OutputConnector> Outputs { get; set; }

            public Dictionary<OutputConnector, Range> Distributions { get; } = new Dictionary<OutputConnector, Range>();
        }

        private class MemoryAllocation
        {
            public MemoryNode Node { get; set; }

            public uint Offset { get; set; }

            public uint Size { get; set; }

            public uint GetAddress() => Node.ValidStart + Offset;

            public uint GetRawAddress() => Node.Start + Offset;

            public MemoryAllocation(MemoryNode memoryNode)
            {
                Node = memoryNode;
                Size = memoryNode.Size;
            }

            public MemoryAllocation(MemoryNode memoryNode, uint offset, uint size)
            {
                Node = memoryNode;
                Offset = offset;
                Size = size;
            }
        }

        private class ConvertContext
        {
            public int InferenceId { get; set; }

            public QuantizationContext Quantization { get; set; }

            public Dictionary<Layer, bool> ProcessMap = new Dictionary<Layer, bool>();

            public Dictionary<Layer, object> LayerArguments { get; } = new Dictionary<Layer, object>();

            public KPUMemoryAllocator KPUMemoryAllocator { get; } = new KPUMemoryAllocator();

            public MainMemoryAllocator MainMemoryAllocator { get; } = new MainMemoryAllocator();

            public List<K210Layer> InferenceOrders { get; } = new List<K210Layer>();

            public Dictionary<OutputConnector, MemoryAllocation> KPUMemoryMap { get; } = new Dictionary<OutputConnector, MemoryAllocation>();

            public Dictionary<OutputConnector, MemoryAllocation> MainMemoryMap { get; } = new Dictionary<OutputConnector, MemoryAllocation>();
        }

        private interface IMemoryAllocator
        {
            void Free(MemoryNode node);
        }

        private class MemoryNode
        {
            private readonly IMemoryAllocator _memoryAllocator;
            private int _useCount;

            public uint Start { get; set; }

            public uint ValidStart
            {
                get
                {
                    if (IsUsed)
                        return Start;
                    else
                        throw new InvalidOperationException("Memory node has been free.");
                }
            }

            public uint Size { get; set; }

            public bool IsUsed => _useCount != 0;

            public MemoryNode(IMemoryAllocator memoryAllocator)
            {
                _memoryAllocator = memoryAllocator;
            }

            public void AddRef()
            {
                _useCount++;
            }

            public void Release()
            {
                if (--_useCount == 0)
                {
                    _memoryAllocator.Free(this);
                }
            }
        }

        private class KPUMemoryAllocator : IMemoryAllocator
        {
            private List<MemoryNode> _nodes = new List<MemoryNode>();

            public uint MaxStart { get; private set; } = 2 * 1024 * 1024 / 64;

            public uint MaxUsage => 2 * 1024 * 1024 / 64 - MaxStart;

            public KPUMemoryAllocator()
            {
                _nodes.Add(new MemoryNode(this) { Start = 0, Size = MaxStart });
            }

            public MemoryNode Allocate(uint size)
            {
                var firstFreeIdx = _nodes.FindLastIndex(o => !o.IsUsed && o.Size >= size);
                if (firstFreeIdx == -1)
                    throw new InvalidOperationException("KPU ran out of memory.");
                var firstFree = _nodes[firstFreeIdx];
                MemoryNode node;
                if (firstFree.Size == size)
                {
                    firstFree.AddRef();
                    node = firstFree;
                }
                else
                {
                    firstFree.Size -= size;
                    var newNode = new MemoryNode(this)
                    {
                        Start = firstFree.Start + firstFree.Size,
                        Size = size
                    };
                    newNode.AddRef();

                    _nodes.Insert(firstFreeIdx + 1, newNode);
                    node = newNode;
                }

                MaxStart = Math.Min(node.Start, MaxStart);
                return node;
            }

            public void Free(MemoryNode node)
            {
                Debug.Assert(!node.IsUsed);
                var idx = _nodes.IndexOf(node);
                if (idx != 0)
                {
                    var before = _nodes[idx - 1];
                    if (!before.IsUsed)
                    {
                        before.Size += node.Size;
                        _nodes.RemoveAt(idx);
                        idx--;
                        node = _nodes[idx];
                    }
                }
                if (idx != _nodes.Count - 1)
                {
                    var after = _nodes[idx + 1];
                    if (!after.IsUsed)
                    {
                        node.Size += after.Size;
                        _nodes.RemoveAt(idx + 1);
                    }
                }
            }
        }

        private class MainMemoryAllocator : IMemoryAllocator
        {
            private List<MemoryNode> _nodes = new List<MemoryNode>();

            public uint MaxEnd { get; private set; }

            public MainMemoryAllocator()
            {
            }

            public MemoryNode Allocate(uint size)
            {
                size = Align(size);
                Reserve(size);
                var firstFreeIdx = _nodes.FindLastIndex(o => !o.IsUsed && o.Size >= size);
                Debug.Assert(firstFreeIdx != -1);
                var firstFree = _nodes[firstFreeIdx];
                MemoryNode node;
                if (firstFree.Size == size)
                {
                    firstFree.AddRef();
                    node = firstFree;
                }
                else
                {
                    firstFree.Size -= size;
                    var newNode = new MemoryNode(this)
                    {
                        Start = firstFree.Start + firstFree.Size,
                        Size = size
                    };
                    newNode.AddRef();

                    _nodes.Insert(firstFreeIdx + 1, newNode);
                    node = newNode;
                }

                return node;
            }

            private uint Align(uint size)
            {
                var remainder = size % 8;
                if (remainder != 0)
                    return size - remainder + 8;
                return size;
            }

            private void Reserve(uint size)
            {
                var firstFreeIdx = _nodes.FindLastIndex(o => !o.IsUsed && o.Size >= size);
                if (firstFreeIdx == -1)
                {
                    if (_nodes.Count == 0 || _nodes.Last().IsUsed)
                    {
                        _nodes.Add(new MemoryNode(this) { Start = MaxEnd, Size = size });
                        MaxEnd += size;
                    }
                    else
                    {
                        var last = _nodes.Last();
                        var toEnlarge = size - last.Size;
                        last.Size += toEnlarge;
                        MaxEnd += toEnlarge;
                    }
                }
            }

            public void Free(MemoryNode node)
            {
                Debug.Assert(!node.IsUsed);
                var idx = _nodes.IndexOf(node);
                if (idx != 0)
                {
                    var before = _nodes[idx - 1];
                    if (!before.IsUsed)
                    {
                        before.Size += node.Size;
                        _nodes.RemoveAt(idx);
                        idx--;
                        node = _nodes[idx];
                    }
                }
                if (idx != _nodes.Count - 1)
                {
                    var after = _nodes[idx + 1];
                    if (!after.IsUsed)
                    {
                        node.Size += after.Size;
                        _nodes.RemoveAt(idx + 1);
                    }
                }
            }
        }
    }
}
