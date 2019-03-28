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

        public uint MainMemoryOutputAddress { get; set; }

        public uint Count { get; set; }

        public K210QuantizationParam QuantParam { get; set; }
    }

    [LayerConverter(typeof(layers.Quantize), K210LayerType.Quantize)]
    public class QuantizeConverter
    {
        public QuantizeLayerArgument Convert(layers.Quantize layer, ConvertContext context)
        {
            return new QuantizeLayerArgument
            {
                Count = (uint)layer.Input.Dimensions.GetSize(),
                QuantParam = context.Quantization.Distributions[layer.Output].Global.GetQuantizationParam(8)
            };
        }

        public void Infer(layers.Quantize layer, QuantizeLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }
    }
}
