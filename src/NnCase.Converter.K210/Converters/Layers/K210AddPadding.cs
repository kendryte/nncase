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
    public class K210AddPaddingLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint KPUMemoryOutputAddress { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(K210AddPadding), K210LayerType.K210AddPadding)]
    public class K210AddPaddingConverter
    {
        public K210AddPaddingLayerArgument Convert(K210AddPadding layer, ConvertContext context)
        {
            return new K210AddPaddingLayerArgument
            {
                Channels = (uint)layer.Input.Dimensions[1]
            };
        }

        public void Infer(K210AddPadding layer, K210AddPaddingLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.KPUMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.None;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.KPUMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public K210AddPaddingLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new K210AddPaddingLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAddress = sr.Read<uint>(),
                KPUMemoryOutputAddress = sr.Read<uint>(),
                Channels = sr.Read<uint>()
            };

            return argument;
        }

        public void Forward(K210AddPaddingLayerArgument argument, ForwardContext context)
        {
            var src = context.GetMainRamAt((int)argument.MainMemoryInputAddress);
            var dest = context.GetKpuRamAt((int)argument.KPUMemoryOutputAddress);

            var height = 4;
            (var groups, var rowLength, var rowPadding) = (4, 1, 16);
            int srcIdx = 0;
            for (int oc = 0; oc < argument.Channels; oc++)
            {
                var channel_origin = oc / groups * rowLength * height * 64 + oc % groups * rowPadding;
                for (int y = 0; y < 1; y++)
                {
                    var y_origin = channel_origin + y * rowLength * 64;
                    for (int x = 0; x < 1; x++)
                        dest[y_origin + x] = src[srcIdx++];
                }
            }
        }
    }
}
