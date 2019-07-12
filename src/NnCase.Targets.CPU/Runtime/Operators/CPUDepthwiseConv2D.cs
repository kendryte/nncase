using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Runtime;

namespace NnCase.Targets.CPU.Runtime.Operators
{
    public class CPUDepthwiseConv2DOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public RuntimeShape InputShape { get; set; }

        public Padding PaddingH { get; set; }

        public Padding PaddingW { get; set; }

        public int FilterH { get; set; }

        public int FilterW { get; set; }

        public int StrideH { get; set; }

        public int StrideW { get; set; }

        public int DilationH { get; set; }

        public int DilationW { get; set; }

        public ValueRange<float> FusedActivation { get; set; }

        public float[] Weights { get; set; }

        public float[] Bias { get; set; }
    }

    public class CPUDepthwiseConv2DOptionsBody : INodeBody
    {
        public RuntimeOpCode OpCode => RuntimeOpCode.CPU_CPUDepthwiseConv2D;

        public CPUDepthwiseConv2DOptions Options { get; set; }

        public void Deserialize(ref MemoryReader reader)
        {
            Options = new CPUDepthwiseConv2DOptions
            {
                Input = reader.Read<MemoryRange>(),
                Output = reader.Read<MemoryRange>(),
                InputShape = reader.Read<RuntimeShape>(),
                PaddingH = reader.Read<Padding>(),
                PaddingW = reader.Read<Padding>(),
                FilterH = reader.Read<int>(),
                FilterW = reader.Read<int>(),
                StrideH = reader.Read<int>(),
                StrideW = reader.Read<int>(),
                DilationH = reader.Read<int>(),
                DilationW = reader.Read<int>(),
                FusedActivation = reader.Read<ValueRange<float>>()
            };

            Options.Weights = reader.ReadArray<float>(Options.InputShape[1] * Options.FilterH * Options.FilterW);
            Options.Bias = reader.ReadArray<float>(Options.InputShape[1]);
        }

        public void Serialize(BinaryWriter writer)
        {
            writer.Write(Options.Input);
            writer.Write(Options.Output);
            writer.Write(Options.InputShape);
            writer.Write(Options.PaddingH);
            writer.Write(Options.PaddingW);
            writer.Write(Options.FilterH);
            writer.Write(Options.FilterW);
            writer.Write(Options.StrideH);
            writer.Write(Options.StrideW);
            writer.Write(Options.DilationH);
            writer.Write(Options.DilationW);
            writer.Write(Options.FusedActivation);
            writer.Write(Options.Weights.AsSpan());
            writer.Write(Options.Bias.AsSpan());
        }
    }
}
