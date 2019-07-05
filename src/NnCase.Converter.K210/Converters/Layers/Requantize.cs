using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Converters.Stages.Quantize;
using NnCase.Converter.K210.Emulator;
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

        public RequantizeLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new RequantizeLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                Count = sr.Read<uint>(),
                Table = sr.ReadArray<byte>(256)
            };

            return argument;
        }

        public void Forward(RequantizeLayerArgument argument, ForwardContext context)
        {
            var src = context.GetMainRamAt((int)argument.MainMemoryInputAddress);
            var dest = context.GetMainRamAt((int)argument.MainMemoryOutputAddress);

            for (int i = 0; i < argument.Count; i++)
                dest[i] = argument.Table[src[i]];
        }
    }
}
