using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;
using NnCase.Evaluation.Operators;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Kernels;
using NnCase.Targets.K210.IR.FakeOperators;
using NnCase.Targets.K210.Kernels;

namespace NnCase.Targets.K210.Evaluation.Operators
{
    internal static partial class K210Evaulators
    {
        private static void RegisterKPUFakeConv2D(EvaluatorRegistry registry)
        {
            registry.Add<KPUFakeConv2D>((n, e) =>
            {
                var input = e.MemoryAt<float>(n.Input);
                var output = e.MemoryAt<float>(n.Output);

                if (n.IsDepthwise)
                    K210Kernels.DepthwiseConv2D(input, output, n.Weights.Buffer.Span, n.Bias.Buffer.Span, OpUtility.To(n.Input.Shape), n.FilterType, n.FusedActivation);
                else
                    K210Kernels.Conv2D(input, output, n.Weights.Buffer.Span, n.Bias.Buffer.Span, OpUtility.To(n.Input.Shape), n.Output.Shape[1], n.FilterType, n.FusedActivation);
            });
        }
    }
}
