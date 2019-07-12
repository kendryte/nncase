using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct ReduceOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public ReduceOperator ReduceOperator { get; set; }

        public RuntimeShape InputShape { get; set; }

        public RuntimeShape OutputShape { get; set; }

        public float InitialValue { get; set; }
    }

    public class ReduceOptionsBody : SimpleNodeBody<ReduceOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.Reduce;
    }
}
