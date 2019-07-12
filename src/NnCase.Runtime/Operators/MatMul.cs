using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public class MatMulOptions
    {
        public MemoryRange InputA { get; set; }

        public MemoryRange InputB { get; set; }

        public MemoryRange Output { get; set; }

        public int ARows { get; set; }

        public int ACols { get; set; }

        public int BCols { get; set; }

        public ValueRange<float> FusedActivation { get; set; }

        public float[] Bias { get; set; }
    }

    public class MatMulOptionsBody : INodeBody
    {
        public RuntimeOpCode OpCode => RuntimeOpCode.MatMul;

        public MatMulOptions Options { get; set; }

        public void Deserialize(ref MemoryReader reader)
        {
            Options = new MatMulOptions
            {
                InputA = reader.Read<MemoryRange>(),
                InputB = reader.Read<MemoryRange>(),
                Output = reader.Read<MemoryRange>(),
                ARows = reader.Read<int>(),
                ACols = reader.Read<int>(),
                BCols = reader.Read<int>(),
                FusedActivation = reader.Read<ValueRange<float>>()
            };

            Options.Bias = reader.ReadArray<float>(Options.BCols);
        }

        public void Serialize(BinaryWriter writer)
        {
            writer.Write(Options.InputA);
            writer.Write(Options.InputB);
            writer.Write(Options.Output);
            writer.Write(Options.ARows);
            writer.Write(Options.ACols);
            writer.Write(Options.BCols);
            writer.Write(Options.FusedActivation);
            writer.Write(Options.Bias.AsSpan());
        }
    }
}
