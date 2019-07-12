using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct PadOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public RuntimePaddings Paddings { get; set; }

        public Scalar PadValue { get; set; }
    }

    public class PadOptionsBody : SimpleNodeBody<PadOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.Pad;
    }
}
