using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using layers = NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
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

    [LayerConverter(typeof(layers.Quantize), K210LayerType.Quantize)]
    public class QuantizeConverter
    {
        public QuantizeLayerArgument Convert(layers.Quantize layer, ConvertContext context)
        {
            return new QuantizeLayerArgument
            {
                Width = (uint)layer.Input.Dimensions[3],
                Height = (uint)layer.Input.Dimensions[2],
                Channels = (uint)layer.Input.Dimensions[1],
                QuantParam = context.Quantization.Distributions[layer.Output].GetQuantizationParam(8)
            };
        }

        public void Infer(layers.Quantize layer, QuantizeLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];

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
        }
    }
}
