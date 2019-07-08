using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Reshape<T>(ReadOnlySpan<T> input, Span<T> output)
        {
            input.CopyTo(output);
        }
    }
}
