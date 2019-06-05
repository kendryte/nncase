using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Converter.K210.Converters.Stages.Convert;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;
using NnCase.Converter.K210.Emulator;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public struct MemoryRange
    {
        public uint Start { get; set; }

        public uint Size { get; set; }
    }

    public class ConcatenationLayerArgument
    {
        public K210LayerFlags Flags { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint InputCount { get; set; }

        public IReadOnlyList<MemoryRange> InputsMainMemory { get; set; }
    }

    [LayerConverter(typeof(Concatenation), K210LayerType.Concatenation)]
    public class ConcatenationConverter
    {
        public ConcatenationLayerArgument Convert(Concatenation layer, ConvertContext context)
        {
            return new ConcatenationLayerArgument
            {
                InputCount = (uint)layer.Inputs.Count
            };
        }

        public void Infer(Concatenation layer, ConcatenationLayerArgument argument, InferenceContext context)
        {
            var outputAlloc = context.MainMemoryMap[layer.Output];
        
            argument.Flags = K210LayerFlags.MainMemoryOutput;
            argument.MainMemoryOutputAddress = outputAlloc.GetAddress();
            argument.InputsMainMemory = (from i in layer.Inputs
                                         let a = context.MainMemoryMap[i.Connection.From]
                                         select new MemoryRange
                                         {
                                             Start = a.GetAddress(),
                                             Size = a.Size
                                         }).ToList();
        }

        public ConcatenationLayerArgument DeserializeBin(int offset, K210BinDeserializeContext context)
        {
            var sr = context.GetReaderAt(offset);
            var argument = new ConcatenationLayerArgument
            {
                Flags = sr.Read<K210LayerFlags>(),
                MainMemoryOutputAddress = sr.Read<uint>(),
                InputCount = sr.Read<uint>()
            };

            argument.InputsMainMemory = sr.ReadArray<MemoryRange>((int)argument.InputCount);
            return argument;
        }

        public void Forward(ConcatenationLayerArgument argument, ForwardContext context)
        {
            var dest = context.GetMainRamAt((int)argument.MainMemoryOutputAddress);

            foreach (var input in argument.InputsMainMemory)
            {
                var src = context.GetMainRamAt((int)input.Start, (int)input.Size);
                src.CopyTo(dest);
                dest = dest.Slice((int)input.Size);
            }
        }
    }
}
