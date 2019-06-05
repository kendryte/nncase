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
    public class L2NormalizationLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(L2Normalization), K210LayerType.L2Normalization)]
    public class L2NormalizationConverter
    {
        public L2NormalizationLayerArgument Convert(L2Normalization layer, ConvertContext context)
        {
            return new L2NormalizationLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
        }

        public void Infer(L2Normalization layer, L2NormalizationLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public L2NormalizationLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new L2NormalizationLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                Channels = sr.Read<uint>()
            };

            return argument;
        }

        public void Forward(L2NormalizationLayerArgument argument, ForwardContext context)
        {
            var src = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAddress));
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            float sum = 0;
            const float epsilon = 1e-10f;
            for (int oc = 0; oc < argument.Channels; oc++)
                sum += src[oc] * src[oc];
            if (sum < epsilon)
                sum = epsilon;
            sum = 1f / (float)Math.Sqrt(sum);

            for (int oc = 0; oc < argument.Channels; oc++)
                dest[oc] = src[oc] * sum;
        }
    }
}
