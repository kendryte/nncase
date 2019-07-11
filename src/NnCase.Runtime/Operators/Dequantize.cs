using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct DequantizeOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public QuantizationParam QuantizationParam { get; set; }
    }

    public class DequantizeOptionsBody : SimpleNodeBody<DequantizeOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.Dequantize;
    }
}
