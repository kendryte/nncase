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
        private static void RegisterConv2D(EvaluatorRegistry registry)
        {
            registry.Add<Conv2D>((n, e) =>
            {
                var input = e.MemoryAt<float>(n.Input);
                var output = e.MemoryAt<float>(n.Output);

                DefaultKernels.Conv2D(input, output, n.Weights.Buffer.Span, n.Bias.Buffer.Span, OpUtility.To(n.Input.Shape), n.Groups, n.Output.Shape[1], n.Weights.Dimensions[2], n.Weights.Dimensions[3], n.StrideH, n.StrideW, n.DilationH, n.DilationW, n.PaddingH, n.PaddingW, n.FusedActivation);
            });
        }
    }
}
