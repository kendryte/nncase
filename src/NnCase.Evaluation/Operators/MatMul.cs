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
        private static void RegisterMatMul(EvaluatorRegistry registry)
        {
            registry.Add<MatMul>((n, e) =>
            {
                var inputA = e.MemoryAt<float>(n.InputA);
                var inputB = e.MemoryAt<float>(n.InputB);
                var output = e.MemoryAt<float>(n.Output);

                DefaultKernels.MatMul(inputA, inputB, output, n.Bias.Buffer.Span, n.InputA.Shape[0], n.InputA.Shape[1], n.InputB.Shape[1], n.FusedActivation);
            });
        }
    }
}
