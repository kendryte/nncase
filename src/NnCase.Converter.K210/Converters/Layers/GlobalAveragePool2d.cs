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
    public class GlobalAveragePool2dLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint KernelSize { get; set; }

        public uint Channels { get; set; }
    }

    [LayerConverter(typeof(GlobalAveragePool), K210LayerType.GlobalAveragePool2d)]
    public class GlobalAveragePool2dConverter
    {
        public GlobalAveragePool2dLayerArgument Convert(GlobalAveragePool layer, ConvertContext context)
        {
            return new GlobalAveragePool2dLayerArgument
            {
                KernelSize = (uint)(layer.Input.Dimensions[2] * layer.Input.Dimensions[3]),
                Channels = (uint)(layer.Input.Dimensions[1])
            };
        }

        public void Infer(GlobalAveragePool layer, GlobalAveragePool2dLayerArgument argument, InferenceContext context)
        {
            var inputAlloc = context.MainMemoryMap[layer.Input.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAddress = inputAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public GlobalAveragePool2dLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new GlobalAveragePool2dLayerArgument();
            argument.Flags = sr.Read<K210LayerFlags>();
            argument.MainMemoryInputAddress = sr.Read<uint>();
            argument.MainMemoryOutputAddress = sr.Read<uint>();
            argument.KernelSize = sr.Read<uint>();
            argument.Channels = sr.Read<uint>();

            return argument;
        }

        public void Forward(GlobalAveragePool2dLayerArgument argument, ForwardContext context)
        {
            var src = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAddress));
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            int i = 0;
            for (int oc = 0; oc < argument.Channels; oc++)
            {
                float sum = 0;
                for (int x = 0; x < argument.KernelSize; x++)
                    sum += src[i++];
                dest[oc] = sum / argument.KernelSize;
            }
        }
    }
}
