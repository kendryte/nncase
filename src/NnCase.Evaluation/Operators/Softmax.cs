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
        private static void RegisterSoftmax(EvaluatorRegistry registry)
        {
            registry.Add<Softmax>((n, e) =>
            {
                var input = e.MemoryAt<float>(n.Input);
                var output = e.MemoryAt<float>(n.Output);

                DefaultKernels.Softmax(input, output, n.Beta, ShapeUtility.ComputeSize(n.Input.Shape) / n.Input.Shape[0], n.Input.Shape[0]);
            });
        }
    }
}
