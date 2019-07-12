using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Pad(int elementSize, ReadOnlySpan<byte> input, Span<byte> output, in RuntimeShape inShape, in RuntimePaddings paddings, in Scalar padValue)
        {
            switch (elementSize)
            {
                case 1:
                    PadImpl(input, output, inShape, paddings, padValue.As<byte>());
                    break;
                case 2:
                    PadImpl(MemoryMarshal.Cast<byte, ushort>(input), MemoryMarshal.Cast<byte, ushort>(output), inShape, paddings, padValue.As<ushort>());
                    break;
                case 4:
                    PadImpl(MemoryMarshal.Cast<byte, uint>(input), MemoryMarshal.Cast<byte, uint>(output), inShape, paddings, padValue.As<uint>());
                    break;
                default:
                    throw ThrowUnsupportedDataType();
            }
        }

        private static void PadImpl<T>(ReadOnlySpan<T> input, Span<T> output, in RuntimeShape inShape, in RuntimePaddings paddings, T padValue)
        {
            var outShape = new RuntimeShape(inShape[0] + paddings[0].Sum, inShape[1] + paddings[1].Sum, inShape[2] + paddings[2].Sum, inShape[3] + paddings[3].Sum);
            int outIdx = 0;

            for (int d0 = 0; d0 < outShape[0]; d0++)
            {
                var d0Origin = -Math.Min(0, paddings[0].Before);
                var in0 = input.Slice((d0Origin + d0) * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int d1 = 0; d1 < outShape[1]; d1++)
                {
                    var d1Origin = -Math.Min(0, paddings[1].Before);
                    var in1 = in0.Slice((d1Origin + d1) * inShape[2] * inShape[3], inShape[2] * inShape[3]);

                    for (int d2 = 0; d2 < outShape[2]; d2++)
                    {
                        var d2Origin = -Math.Min(0, paddings[2].Before);
                        var in2 = in1.Slice((d2Origin + d2) * inShape[3], inShape[3]);

                        for (int d3 = 0; d3 < outShape[3]; d3++)
                        {
                            var d3Origin = -Math.Min(0, paddings[3].Before);

                            if (d0 < paddings[0].Before || d0 >= outShape[0] - paddings[0].After
                                || d1 < paddings[1].Before || d1 >= outShape[1] - paddings[1].After
                                || d2 < paddings[2].Before || d2 >= outShape[2] - paddings[2].After
                                || d3 < paddings[3].Before || d1 >= outShape[3] - paddings[3].After)
                                output[outIdx++] = padValue;
                            else
                                output[outIdx++] = in2[d3Origin + d3];
                        }
                    }
                }
            }
        }
    }
}
