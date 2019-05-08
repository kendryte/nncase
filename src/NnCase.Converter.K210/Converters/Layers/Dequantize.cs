using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Emulator;
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

        public DequantizeLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new DequantizeLayerArgument();
            argument.Flags = sr.Read<K210LayerFlags>();
            argument.MainMemoryInputAddress = sr.Read<uint>();
            argument.MainMemoryOutputAddress = sr.Read<uint>();
            argument.Count = sr.Read<uint>();
            argument.QuantParam = sr.Read<K210QuantizationParam>();

            return argument;
        }

        public void Forward(DequantizeLayerArgument argument, ForwardContext context)
        {
            var src = context.GetMainRamAt((int)argument.MainMemoryInputAddress);
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));
            var q = argument.QuantParam;

            for (int i = 0; i < argument.Count; i++)
                dest[i] = src[i] * q.Scale + q.Bias;
        }
    }
}
