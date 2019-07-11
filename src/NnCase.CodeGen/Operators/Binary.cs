using System;
using System.Collections.Generic;
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
        private static void RegisterBinary(CodeGenRegistry registry)
        {
            registry.Add<Binary>((n, g) =>
            {
                return new BinaryOptionsBody
                {
                    Options = new BinaryOptions
                    {
                        InputA = g.MemoryRange(n.InputA),
                        InputB = g.MemoryRange(n.InputB),
                        Output = g.MemoryRange(n.Output),
                        BinaryOperator = n.BinaryOperator,
                        InputAShape = OpUtility.To(n.InputA.Shape),
                        InputBShape = OpUtility.To(n.InputB.Shape),
                        OutputShape = OpUtility.To(n.Output.Shape),
                        FusedActivation = n.FusedActivation
                    }
                };
            });
        }
    }
}
