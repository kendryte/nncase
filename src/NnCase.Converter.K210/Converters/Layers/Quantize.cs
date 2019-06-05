using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Emulator;
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

        public QuantizeLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new QuantizeLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                Count = sr.Read<uint>(),
                QuantParam = sr.Read<K210QuantizationParam>()
            };

            return argument;
        }

        public void Forward(QuantizeLayerArgument argument, ForwardContext context)
        {
            var src = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAddress));
            var dest = context.GetMainRamAt((int)argument.MainMemoryOutputAddress);
            var q = argument.QuantParam;
            float scale = 1f / q.Scale;

            for (int i = 0; i < argument.Count; i++)
            {
                int value = (int)Math.Round((src[i] - q.Bias) * scale);
                dest[i] = (byte)FxExtensions.Clamp(value, 0, 0xFF);
            }
        }
    }
}
