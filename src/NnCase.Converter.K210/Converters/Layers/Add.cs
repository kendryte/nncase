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
    public class AddLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryInputAAddress { get; set; }

        public uint MainMemoryInputBAddress { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint Count { get; set; }
    }

    [LayerConverter(typeof(Add), K210LayerType.Add)]
    public class AddConverter
    {
        public AddLayerArgument Convert(Add layer, ConvertContext context)
        {
            return new AddLayerArgument
            {
                Count = (uint)(layer.Output.Dimensions.GetSize())
            };
        }

        public void Infer(Add layer, AddLayerArgument argument, InferenceContext context)
        {
            var inputAAlloc = context.MainMemoryMap[layer.InputA.Connection.From];
            var inputBAlloc = context.MainMemoryMap[layer.InputB.Connection.From];
            var outputAlloc = context.MainMemoryMap[layer.Output];

            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryInputAAddress = inputAAlloc.GetAddress();
            argument.MainMemoryInputBAddress = inputBAlloc.GetAddress();
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
        }

        public AddLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new AddLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryInputAAddress = sr.Read<uint>(),
                MainMemoryInputBAddress = sr.Read<uint>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                Count = sr.Read<uint>()
            };

            return argument;
        }

        public void Forward(AddLayerArgument argument, ForwardContext context)
        {
            var srcA = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputAAddress));
            var srcB = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryInputBAddress));
            var dest = MemoryMarshal.Cast<byte, float>(context.GetMainRamAt((int)argument.MainMemoryOutputAddress));

            for (int oc = 0; oc < argument.Count; oc++)
                dest[oc] = srcA[oc] + srcB[oc];
        }
    }
}
