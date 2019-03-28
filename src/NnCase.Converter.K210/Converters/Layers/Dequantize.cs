using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class DequantizeLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Count { get; set; }

        public K210QuantizationParam QuantParam { get; set; }
    }

    [LayerConverter(typeof(Dequantize), K210LayerType.Dequantize)]
    public class DequantizeConverter
    {
        public DequantizeLayerArgument Convert(Dequantize layer, ConvertContext context)
        {
            return new DequantizeLayerArgument
            {
                Count = (uint)(layer.Input.Dimensions.GetSize()),
                QuantParam = context.Quantization.Distributions[layer.Input.Connection.From].Global.GetQuantizationParam(8)
            };
        }

        public void Infer(Dequantize layer, DequantizeLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
