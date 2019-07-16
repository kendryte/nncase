using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct ResizeBilinearOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public RuntimeShape InputShape { get; set; }

        public int OutputWidth { get; set; }

        public int OutputHeight { get; set; }

        public bool AlignCorners { get; set; }
    }

    public class ResizeBilinearOptionsBody : SimpleNodeBody<ResizeBilinearOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.ResizeBilinear;
    }
}
