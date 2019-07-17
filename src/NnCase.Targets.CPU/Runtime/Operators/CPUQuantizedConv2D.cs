using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Runtime;

namespace NnCase.Targets.CPU.Runtime.Operators
{
    public class CPUQuantizedConv2DOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public RuntimeShape InputShape { get; set; }

        public int OutputChannels { get; set; }

        public Padding PaddingH { get; set; }

        public Padding PaddingW { get; set; }

        public int FilterH { get; set; }

        public int FilterW { get; set; }

        public int StrideH { get; set; }

        public int StrideW { get; set; }

        public int DilationH { get; set; }

        public int DilationW { get; set; }

        public int InputOffset { get; set; }

        public int FilterOffset { get; set; }

        public int OutputMul { get; set; }

        public int OutputShift { get; set; }

        public int OutputOffset { get; set; }

        public byte[] Weights { get; set; }

        public int[] Bias { get; set; }
    }

    public class CPUQuantizedConv2DOptionsBody : INodeBody
    {
        public RuntimeOpCode OpCode => RuntimeOpCode.CPU_CPUQuantizedConv2D;

        public CPUQuantizedConv2DOptions Options { get; set; }

        public void Deserialize(ref MemoryReader reader)
        {
            Options = new CPUQuantizedConv2DOptions
            {
                Input = reader.Read<MemoryRange>(),
                Output = reader.Read<MemoryRange>(),
                InputShape = reader.Read<RuntimeShape>(),
                OutputChannels = reader.Read<int>(),
                PaddingH = reader.Read<Padding>(),
                PaddingW = reader.Read<Padding>(),
                FilterH = reader.Read<int>(),
                FilterW = reader.Read<int>(),
                StrideH = reader.Read<int>(),
                StrideW = reader.Read<int>(),
                DilationH = reader.Read<int>(),
                DilationW = reader.Read<int>(),
                InputOffset = reader.Read<int>(),
                FilterOffset = reader.Read<int>(),
                OutputMul = reader.Read<int>(),
                OutputShift = reader.Read<int>(),
                OutputOffset = reader.Read<int>()
            };

            Options.Weights = reader.ReadArray<byte>(Options.OutputChannels * Options.InputShape[3] * Options.FilterH * Options.FilterW);
            Options.Bias = reader.ReadArray<int>(Options.OutputChannels);
        }

        public void Serialize(BinaryWriter writer)
        {
            writer.Write(Options.Input);
            writer.Write(Options.Output);
            writer.Write(Options.InputShape);
            writer.Write(Options.OutputChannels);
            writer.Write(Options.PaddingH);
            writer.Write(Options.PaddingW);
            writer.Write(Options.FilterH);
            writer.Write(Options.FilterW);
            writer.Write(Options.StrideH);
            writer.Write(Options.StrideW);
            writer.Write(Options.DilationH);
            writer.Write(Options.DilationW);
            writer.Write(Options.InputOffset);
            writer.Write(Options.FilterOffset);
            writer.Write(Options.OutputMul);
            writer.Write(Options.OutputShift);
            writer.Write(Options.OutputOffset);
            writer.Write(Options.Weights.AsSpan());
            writer.Write(Options.Bias.AsSpan());
        }
    }
}
