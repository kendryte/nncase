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

                var convOutShape = n.Input.Shape.Clone();
                convOutShape[1] = n.Output.Shape[1];
                var workspace = new float[ShapeUtility.ComputeSize(convOutShape)];

                if (n.IsDepthwise)
                    K210Kernels.DepthwiseConv2D(input, workspace, n.Weights.Buffer.Span, n.Bias.Buffer.Span, OpUtility.To(n.Input.Shape), n.FilterType, n.FusedActivation);
                else
                    K210Kernels.Conv2D(input, workspace, n.Weights.Buffer.Span, n.Bias.Buffer.Span, OpUtility.To(n.Input.Shape), n.Output.Shape[1], n.FilterType, n.FusedActivation);
                K210Kernels.Pool2D(workspace, output, OpUtility.To(convOutShape), n.PoolType);
            });
        }
    }
}
