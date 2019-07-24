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
        private static void RegisterStridedSlice(EvaluatorRegistry registry)
        {
            registry.Add<StridedSlice>((n, e) =>
            {
                var input = e.MemoryAt<byte>(n.Input);
                var output = e.MemoryAt<byte>(n.Output);

                var elementSize = ShapeUtility.GetBytes(n.Input.Type);
                DefaultKernels.StridedSlice(elementSize, input, output, OpUtility.To(n.Input.Shape), OpUtility.To(n.Begin), OpUtility.To(n.End), OpUtility.To(n.Strides));
            });
        }
    }
}
