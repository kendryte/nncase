using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Emulator;
using NnCase.Converter.K210.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class K210RemovePaddingLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(K210RemovePadding), K210LayerType.K210RemovePadding)]
    public class K210RemovePaddingConverter
    {
        public K210RemovePaddingLayerArgument Convert(K210RemovePadding layer, ConvertContext context)
        {
            return new K210RemovePaddingLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
        }

        public void Infer(K210RemovePadding layer, K210RemovePaddingLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public K210RemovePaddingLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new K210RemovePaddingLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                Channels = sr.Read<uint>()
            };

            return argument;
        }

        public void Forward(K210RemovePaddingLayerArgument argument, ForwardContext context)
        {
            var src = context.GetMainRamAt((int)argument.MainMemoryInputAddress);
            var dest = context.GetMainRamAt((int)argument.MainMemoryOutputAddress);

            for (int oc = 0; oc < argument.Channels; oc++)
                dest[oc] = src[oc * 16];
        }
    }
}
