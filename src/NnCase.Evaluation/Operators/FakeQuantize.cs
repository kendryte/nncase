using System;
using System.Collections.Generic;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Kernels;

namespace NnCase.Evaluation.Operators
{
    internal static partial class DefaultEvaulators
    {
        private static void RegisterFakeQuantize(EvaluatorRegistry registry)
        {
            registry.Add<FakeQuantize>((n, e) =>
            {
                var input = e.MemoryAt<byte>(n.Input);
                var output = e.MemoryAt<byte>(n.Output);

                DefaultKernels.MemoryCopy(input, output);
            });

            registry.Add<FakeDequantize>((n, e) =>
            {
                var input = e.MemoryAt<byte>(n.Input);
                var output = e.MemoryAt<byte>(n.Output);

                DefaultKernels.MemoryCopy(input, output);
            });
        }
    }
}
