using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class RequantizeLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Count { get; set; }

        public byte[] Table { get; set; }
    }

    [LayerConverter(typeof(Requantize), K210LayerType.Requantize)]
    public class RequantizeConverter
    {
        public RequantizeLayerArgument Convert(Requantize layer, ConvertContext context)
        {
            var ir = context.Quantization.Distributions[layer.Input.Connection.From].Global;
            var or = context.Quantization.Distributions[layer.Output].Global;

            return new RequantizeLayerArgument
            {
                Count = (uint)layer.Input.Dimensions.GetSize(),
                Table = Quantizer.GetRequantizeTable(ir, or)
            };
        }

        public void Infer(Requantize layer, RequantizeLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
