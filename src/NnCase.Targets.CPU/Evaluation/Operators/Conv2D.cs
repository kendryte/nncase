using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;
using NnCase.Evaluation.Operators;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Kernels;
using NnCase.Targets.CPU.IR.Operators;
using NnCase.Targets.CPU.Kernels;

namespace NnCase.Targets.CPU.Evaluation.Operators
{
    internal static partial class CPUEvaulators
    {
        private static void RegisterCPUConv2D(EvaluatorRegistry registry)
        {
            registry.Add<CPUConv2D>((n, e) =>
            {
                var input = e.MemoryAt<float>(n.Input);
                var output = e.MemoryAt<float>(n.Output);

                CPUKernels.Conv2D(input, output, n.Weights.Buffer.Span, n.Bias.Buffer.Span, OpUtility.To(n.Input.Shape), n.Output.Shape[3], n.Weights.Dimensions[1], n.Weights.Dimensions[2], n.StrideH, n.StrideW, n.DilationH, n.DilationW, n.PaddingH, n.PaddingW, n.FusedActivation);
            });
        }

        private static void RegisterCPUDepthwiseConv2D(EvaluatorRegistry registry)
        {
            registry.Add<CPUDepthwiseConv2D>((n, e) =>
            {
                var input = e.MemoryAt<float>(n.Input);
                var output = e.MemoryAt<float>(n.Output);

                CPUKernels.DepthwiseConv2D(input, output, n.Weights.Buffer.Span, n.Bias.Buffer.Span, OpUtility.To(n.Input.Shape), n.Weights.Dimensions[1], n.Weights.Dimensions[2], n.StrideH, n.StrideW, n.DilationH, n.DilationW, n.PaddingH, n.PaddingW, n.FusedActivation);
            });
        }
    }
}
