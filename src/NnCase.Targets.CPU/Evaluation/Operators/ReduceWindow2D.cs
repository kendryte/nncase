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
        private static void RegisterCPUReduceWindow2D(EvaluatorRegistry registry)
        {
            registry.Add<CPUReduceWindow2D>((n, e) =>
            {
                void ReduceWindow2D(Func<float, float, float> binaryOp, Func<float, int, float> windowOp)
                {
                    var input = e.MemoryAt<float>(n.Input);
                    var output = e.MemoryAt<float>(n.Output);

                    CPUKernels.ReduceWindow2D(input, output, n.InitialValue, OpUtility.To(n.Input.Shape), n.FilterH, n.FilterW, n.StrideH, n.StrideW, n.DilationH, n.DilationW, n.PaddingH, n.PaddingW, n.FusedActivation, binaryOp, windowOp);
                }

                switch (n.ReduceOperator)
                {
                    case ReduceOperator.Max:
                        ReduceWindow2D((a, b) => Math.Max(a, b), (v, k) => v);
                        break;
                    case ReduceOperator.Mean:
                        ReduceWindow2D((a, b) => a + b, (v, k) => v / k);
                        break;
                    case ReduceOperator.Min:
                        ReduceWindow2D((a, b) => Math.Min(a, b), (v, k) => v);
                        break;
                    default:
                        throw new NotSupportedException($"Unsupported reduce operator: {n.ReduceOperator}");
                }
            });
        }
    }
}
