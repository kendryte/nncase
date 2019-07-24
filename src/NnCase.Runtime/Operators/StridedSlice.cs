using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public struct StridedSliceOptions
    {
        public MemoryRange Input { get; set; }

        public MemoryRange Output { get; set; }

        public RuntimeShape InputShape { get; set; }

        public RuntimeShape Begin { get; set; }

        public RuntimeShape End { get; set; }

        public RuntimeShape Strides { get; set; }

        public int BeginMask { get; set; }

        public int EndMask { get; set; }

        public int EllipsisMask { get; set; }

        public int NewAxisMask { get; set; }

        public int ShrinkAxisMask { get; set; }
    }

    public class StridedSliceOptionsBody : SimpleNodeBody<StridedSliceOptions>
    {
        public override RuntimeOpCode OpCode => RuntimeOpCode.StridedSlice;
    }
}
