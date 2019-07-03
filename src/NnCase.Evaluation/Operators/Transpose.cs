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
        private static void RegisterTranspose(EvaluatorRegistry registry)
        {
            registry.Add<Transpose>((n, e) =>
            {
                var input = e.MemoryAt<byte>(n.Input);
                var output = e.MemoryAt<byte>(n.Output);

                var elementSize = ShapeUtility.GetBytes(n.Input.Type);
                (var rtInShape, var rtPerm) = OpUtility.ExtendTransposeShape(n.Input.Shape, n.Perm);
                DefaultKernels.Transpose(elementSize, input, output, rtInShape, rtPerm);
            });
        }
    }
}
