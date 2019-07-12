using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Evaluation;
using NnCase.Evaluation.Operators;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Runtime.Operators;

namespace NnCase.CodeGen.Operators
{
    internal static partial class DefaultEmitters
    {
        private static void RegisterMatMul(CodeGenRegistry registry)
        {
            registry.Add<MatMul>((n, g) =>
            {
                return new MatMulOptionsBody
                {
                    Options = new MatMulOptions
                    {
                        InputA = g.MemoryRange(n.InputA),
                        InputB = g.MemoryRange(n.InputB),
                        Output = g.MemoryRange(n.Output),
                        ARows = n.InputA.Shape[0],
                        ACols = n.InputA.Shape[1],
                        BCols = n.InputB.Shape[1],
                        FusedActivation = n.FusedActivation,
                        Bias = n.Bias.Buffer.ToArray()
                    }
                };
            });
        }
    }
}
