using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct QuantizeOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public QuantizationParam QuantizationParam { get; set; }
    }

    public class QuantizeOptionsBody : SimpleNodeBody<QuantizeOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.Quantize;
    }
}
