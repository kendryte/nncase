using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void StridedSlice(int elementSize, ReadOnlySpan<byte> input, Span<byte> output, in RuntimeShape inShape, in RuntimeShape begin, in RuntimeShape end, in RuntimeShape strides)
        {
            switch (elementSize)
            {
                case 1:
                    StridedSliceImpl(input, output, inShape, begin, end, strides);
                    break;
                case 2:
                    StridedSliceImpl(MemoryMarshal.Cast<byte, ushort>(input), MemoryMarshal.Cast<byte, ushort>(output), inShape, begin, end, strides);
                    break;
                case 4:
                    StridedSliceImpl(MemoryMarshal.Cast<byte, uint>(input), MemoryMarshal.Cast<byte, uint>(output), inShape, begin, end, strides);
                    break;
                default:
                    throw ThrowUnsupportedDataType();
            }
        }

        private static void StridedSliceImpl<T>(ReadOnlySpan<T> input, Span<T> output, in RuntimeShape inShape, in RuntimeShape begin, in RuntimeShape end, in RuntimeShape strides)
        {
            bool LoopCondition(int i, int stop, int stride)
            {
                return stride > 0 ? i < stop : i > stop;
            }

            int outIdx = 0;
            for (int d0 = begin[0]; LoopCondition(d0, end[0], strides[0]); d0 += strides[0])
            {
                var d0Origin = input.Slice(d0 * inShape[1] * inShape[2] * inShape[3]);
                for (int d1 = begin[1]; LoopCondition(d1, end[1], strides[1]); d1 += strides[1])
                {
                    var d1Origin = d0Origin.Slice(d1 * inShape[2] * inShape[3]);
                    for (int d2 = begin[2]; LoopCondition(d2, end[2], strides[2]); d2 += strides[2])
                    {
                        var d2Origin = d1Origin.Slice(d2 * inShape[3]);
                        for (int d3 = begin[3]; LoopCondition(d3, end[3], strides[3]); d3 += strides[3])
                        {
                            output[outIdx++] = d2Origin[d3];
                        }
                    }
                }
            }
        }
    }
}
