using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void ResizeNearestNeighbor(int elementSize, ReadOnlySpan<byte> input, Span<byte> output, in RuntimeShape inShape, int outputHeight, int outputWidth)
        {
            switch (elementSize)
            {
                case 1:
                    ResizeNearestNeighborImpl(input, output, inShape, outputHeight, outputWidth);
                    break;
                case 2:
                    ResizeNearestNeighborImpl(MemoryMarshal.Cast<byte, ushort>(input), MemoryMarshal.Cast<byte, ushort>(output), inShape, outputHeight, outputWidth);
                    break;
                case 4:
                    ResizeNearestNeighborImpl(MemoryMarshal.Cast<byte, uint>(input), MemoryMarshal.Cast<byte, uint>(output), inShape, outputHeight, outputWidth);
                    break;
                default:
                    throw ThrowUnsupportedDataType();
            }
        }

        private static void ResizeNearestNeighborImpl<T>(ReadOnlySpan<T> input, Span<T> output, in RuntimeShape inShape, int outputHeight, int outputWidth)
        {
            var heightScale = (float)inShape[2] / outputHeight;
            var widthScale = (float)inShape[3] / outputWidth;

            var destIdx = 0;
            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int oc = 0; oc < inShape[1]; oc++)
                {
                    var inC = inBatch.Slice(oc * inShape[2] * inShape[3], inShape[2] * inShape[3]);

                    for (int oy = 0; oy < outputHeight; oy++)
                    {
                        var inY = (int)Math.Min(MathF.Floor(oy * heightScale), inShape[2] - 1);
                        var inRow = inC.Slice(inY * inShape[3], inShape[3]);

                        for (int ox = 0; ox < outputWidth; ox++)
                        {
                            var inX = (int)Math.Min(MathF.Floor(ox * widthScale), inShape[3] - 1);
                            output[destIdx++] = inRow[inX];
                        }
                    }
                }
            }
        }

        public static void ResizeBilinear(ReadOnlySpan<float> input, Span<float> output, in RuntimeShape inShape, int outputHeight, int outputWidth, bool alignCorners)
        {
            var heightScale = (float)inShape[2] / outputHeight;
            var widthScale = (float)inShape[3] / outputWidth;
            if (alignCorners && outputHeight > 1)
                heightScale = (float)(inShape[2] - 1) / (outputHeight - 1);
            if (alignCorners && outputWidth > 1)
                widthScale = (float)(inShape[3] - 1) / (outputWidth - 1);

            var destIdx = 0;
            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int oc = 0; oc < inShape[1]; oc++)
                {
                    var inC = inBatch.Slice(oc * inShape[2] * inShape[3], inShape[2] * inShape[3]);

                    for (int oy = 0; oy < outputHeight; oy++)
                    {
                        var inY = oy * heightScale;
                        var inY0 = (int)MathF.Floor(inY);
                        var inY1 = Math.Min(inY0 + 1, inShape[2] - 1);

                        for (int ox = 0; ox < outputWidth; ox++)
                        {
                            var inX = ox * widthScale;
                            var inX0 = (int)MathF.Floor(inX);
                            var inX1 = Math.Min(inX0 + 1, inShape[3] - 1);

                            var v0 = inC[inY0 * inShape[3] + inX0];
                            var v1 = inC[inY1 * inShape[3] + inX0];
                            var v2 = inC[inY0 * inShape[3] + inX1];
                            var v3 = inC[inY1 * inShape[3] + inX1];

                            var a0 = (1 - (inY - inY0)) * (1 - (inX - inX0));
                            var a1 = (inY - inY0) * (1 - (inX - inX0));
                            var a2 = (1 - (inY - inY0)) * (inX - inX0);
                            var a3 = (inY - inY0) * (inX - inX0);

                            output[destIdx++] = v0 * a0 + v1 * a1 + v2 * a2 + v3 * a3;
                        }
                    }
                }
            }
        }
    }
}
