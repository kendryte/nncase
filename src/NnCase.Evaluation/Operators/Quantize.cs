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
        private static void RegisterQuantize(EvaluatorRegistry registry)
        {
            registry.Add<Quantize>((n, e) =>
            {
                var input = e.MemoryAt<float>(n.Input);
                var output = e.MemoryAt<byte>(n.Output);

                DefaultKernels.Quantize(input, output, n.QuantizationParam);
            });

            registry.Add<Dequantize>((n, e) =>
            {
                var input = e.MemoryAt<byte>(n.Input);
                var output = e.MemoryAt<float>(n.Output);

                DefaultKernels.Dequantize(input, output, n.QuantizationParam);
            });
        }
    }
}
