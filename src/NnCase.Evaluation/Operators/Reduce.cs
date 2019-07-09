using System;
using System.Collections.Generic;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Kernels;
using static LanguageExt.Prelude;

namespace NnCase.Evaluation.Operators
{
    internal static partial class DefaultEvaulators
    {
        private static void RegisterReduce(EvaluatorRegistry registry)
        {
            registry.Add<Reduce>((n, e) =>
            {
                void Reduce(Func<float, float, float> binaryOp)
                {
                    var input = e.MemoryAt<float>(n.Input);
                    var output = e.MemoryAt<float>(n.Output);

                    var reducedShape = ShapeUtility.GetReducedShape(n.Input.Shape, n.Axis, true);
                    DefaultKernels.Reduce(input, output, OpUtility.To(n.Input.Shape), OpUtility.To(reducedShape), n.InitialValue, binaryOp);
                }

                switch (n.ReduceOperator)
                {
                    case ReduceOperator.Max:
                        Reduce((a, b) => Math.Max(a, b));
                        break;
                    case ReduceOperator.Mean:
                        {
                            Reduce((a, b) => a + b);
                            var divider = (float)ShapeUtility.ComputeSize(n.Input.Shape) / ShapeUtility.ComputeSize(n.Output.Shape);
                            var memory = e.MemoryAt<float>(n.Output);
                            DefaultKernels.Unary(memory, memory, x => x / divider);
                        break;
                        }
                    case ReduceOperator.Min:
                        Reduce((a, b) => Math.Min(a, b));
                        break;
                    default:
                        throw new NotSupportedException($"Unsupported reduce operator: {n.ReduceOperator}");
                }
            });
        }
    }
}
