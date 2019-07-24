using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR.Operators
{
    public class StridedSlice : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public Shape Begin { get; }

        public Shape End { get; }

        public Shape Strides { get; }

        public int BeginMask { get; }

        public int EndMask { get; }

        public int EllipsisMask { get; }

        public int NewAxisMask { get; }

        public int ShrinkAxisMask { get; }

        public StridedSlice(DataType type, Shape inputShape, Shape begin, Shape end, Shape strides, int beginMask, int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask)
        {
            Begin = ShapeUtility.NormalizeStridedSliceBegin(inputShape, begin, strides, beginMask);
            End = ShapeUtility.NormalizeStridedSliceEnd(inputShape, Begin, end, strides, endMask, shrinkAxisMask);
            Strides = strides;
            BeginMask = 0;
            EndMask = 0;
            EllipsisMask = ellipsisMask;
            NewAxisMask = newAxisMask;
            ShrinkAxisMask = shrinkAxisMask;

            Input = AddInput("input", type, inputShape);
            Output = AddOutput("output", type, ShapeUtility.GetStridedSliceOutputShape(Begin, End, strides, ellipsisMask, newAxisMask, shrinkAxisMask));
        }
    }
}
