using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct TransposeOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public RuntimeShape InputShape { get; set; }

        public RuntimeShape Perm { get; set; }

        public int ElementSize { get; set; }
    }

    public class TransposeOptionsBody : SimpleNodeBody<TransposeOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.Transpose;
    }
}
