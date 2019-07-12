using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct MemoryCopyOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }
    }

    public class MemoryCopyOptionsBody : SimpleNodeBody<MemoryCopyOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.MemoryCopy;
    }
}
