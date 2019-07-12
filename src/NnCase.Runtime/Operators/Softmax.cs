using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct SoftmaxOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public int InnerSize { get; set; }

        public int OuterSize { get; set; }

        public float Beta { get; set; }
    }

    public class SoftmaxOptionsBody : SimpleNodeBody<SoftmaxOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.Softmax;
    }
}
