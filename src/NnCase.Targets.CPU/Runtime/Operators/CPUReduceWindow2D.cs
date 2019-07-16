using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct CPUReduceWindow2DOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public ReduceOperator ReduceOperator { get; set; }

        public RuntimeShape InputShape { get; set; }

        public Padding PaddingH { get; set; }

        public Padding PaddingW { get; set; }

        public int FilterH { get; set; }

        public int FilterW { get; set; }

        public int StrideH { get; set; }

        public int StrideW { get; set; }

        public int DilationH { get; set; }

        public int DilationW { get; set; }

        public float InitialValue { get; set; }

        public ValueRange<float> FusedActivation { get; set; }
    }

    public class CPUReduceWindow2DOptionsBody : SimpleNodeBody<CPUReduceWindow2DOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.CPU_CPUReduceWindow2D;
    }
}
