using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct ResizeNearestNeighborOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public int OutputWidth { get; set; }

        public int OutputHeight { get; set; }

        public int ElementSize { get; set; }

        public bool AlignCorners { get; set; }
    }

    public class ResizeNearestNeighborOptionsBody : SimpleNodeBody<ResizeNearestNeighborOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.ResizeNearestNeighbor;
    }
}
