using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public class ConcatOptions
    {
        public MemoryRange Output { get; set; }

        public int InnerSize { get; set; }

        public int OuterSize { get; set; }

        public int InputsCount { get; set; }

        public MemoryRange[] Inputs { get; set; }

        public int[] Dimensions { get; set; }
    }

    public class ConcatOptionsBody : INodeBody
    {
        public RuntimeOpCode OpCode => RuntimeOpCode.Concat;

        public ConcatOptions Options { get; set; }

        public void Deserialize(ref MemoryReader reader)
        {
            Options = new ConcatOptions
            {
                Output = reader.Read<MemoryRange>(),
                InnerSize = reader.Read<int>(),
                OuterSize = reader.Read<int>(),
                InputsCount = reader.Read<int>()
            };

            Options.Inputs = reader.ReadArray<MemoryRange>(Options.InputsCount);
            Options.Dimensions = reader.ReadArray<int>(Options.InputsCount);
        }

        public void Serialize(BinaryWriter writer)
        {
            writer.Write(Options.Output);
            writer.Write(Options.InnerSize);
            writer.Write(Options.OuterSize);
            writer.Write(Options.InputsCount);
            writer.Write(Options.Inputs.AsSpan());
            writer.Write(Options.Dimensions.AsSpan());
        }
    }
}
