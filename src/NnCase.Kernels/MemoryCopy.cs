using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void MemoryCopy(ReadOnlySpan<byte> input, Span<byte> output)
        {
            input.CopyTo(output);
        }
    }
}
