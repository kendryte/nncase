using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Concat(IReadOnlyList<ReadOnlyMemory<byte>> inputs, Span<byte> output, IReadOnlyList<int> concatDimensions, int innerSize, int outerSize)
        {
            var outIdx = 0;
            for (int oc = 0; oc < outerSize; oc++)
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    var size = innerSize * concatDimensions[i];
                    var src = inputs[i].Span.Slice(oc * size, size);
                    var dest = output.Slice(outIdx, size);
                    src.CopyTo(dest);
                    outIdx += size;
                }
            }
        }
    }
}
