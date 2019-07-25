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
        private static void RegisterBinary(EvaluatorRegistry registry)
        {
            registry.Add<Binary>((n, e) =>
            {
                void Binary(Func<float, float, float> binaryOp)
                {
                    var inputA = e.MemoryAt<float>(n.InputA);
                    var inputB = e.MemoryAt<float>(n.InputB);
                    var output = e.MemoryAt<float>(n.Output);

                    DefaultKernels.Binary(inputA, inputB, output, OpUtility.To(n.InputA.Shape), OpUtility.To(n.InputB.Shape), OpUtility.To(n.Output.Shape), n.FusedActivation, binaryOp);
                }

                switch (n.BinaryOperator)
                {
                    case BinaryOperator.Add:
                        Binary((a, b) => a + b);
                        break;
                    case BinaryOperator.Sub:
                        Binary((a, b) => a - b);
                        break;
                    case BinaryOperator.Mul:
                        Binary((a, b) => a * b);
                        break;
                    case BinaryOperator.Div:
                        Binary((a, b) => a / b);
                        break;
                    case BinaryOperator.Min:
                        Binary((a, b) => Math.Min(a, b));
                        break;
                    case BinaryOperator.Max:
                        Binary((a, b) => Math.Max(a, b));
                        break;
                    default:
                        throw new NotSupportedException($"Unsupported binary operator: {n.BinaryOperator}");
                }
            });
        }
    }
}
