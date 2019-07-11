using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct BinaryOptions
    {
        public MemoryRange InputA { get; set; }

        public MemoryRange InputB { get; set; }

        public MemoryRange Output { get; set; }

        public BinaryOperator BinaryOperator { get; set; }

        public RuntimeShape InputAShape { get; set; }

        public RuntimeShape InputBShape { get; set; }

        public RuntimeShape OutputShape { get; set; }

        public ValueRange<float> FusedActivation { get; set; }
    }

    public class BinaryOptionsBody : SimpleNodeBody<BinaryOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.Binary;
    }
}
