using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Transpose(int elementSize, ReadOnlySpan<byte> input, Span<byte> output, in RuntimeShape inShape, in RuntimeShape perm)
        {
            switch (elementSize)
            {
                case 1:
                    TransposeImpl(input, output, inShape, perm);
                    break;
                case 2:
                    TransposeImpl(MemoryMarshal.Cast<byte, ushort>(input), MemoryMarshal.Cast<byte, ushort>(output), inShape, perm);
                    break;
                case 4:
                    TransposeImpl(MemoryMarshal.Cast<byte, uint>(input), MemoryMarshal.Cast<byte, uint>(output), inShape, perm);
                    break;
                default:
                    throw ThrowUnsupportedDataType();
            }
        }

        private static void TransposeImpl<T>(ReadOnlySpan<T> input, Span<T> output, in RuntimeShape inShape, in RuntimeShape perm)
        {
            RuntimeShape outShape;
            for (int idx = 0; idx < 4; idx++)
                outShape[idx] = inShape[perm[idx]];

            RuntimeShape i, o;
            for (o[3] = 0; o[3] < outShape[3]; o[3]++)
            {
                i[perm[3]] = o[3];
                for (o[2] = 0; o[2] < outShape[2]; o[2]++)
                {
                    i[perm[2]] = o[2];
                    for (o[1] = 0; o[1] < outShape[1]; o[1]++)
                    {
                        i[perm[1]] = o[1];
                        for (o[0] = 0; o[0] < outShape[0]; o[0]++)
                        {
                            i[perm[0]] = o[0];
                            output[Offset(outShape, o)] = input[Offset(inShape, i)];
                        }
                    }
                }
            }
        }
    }
}
