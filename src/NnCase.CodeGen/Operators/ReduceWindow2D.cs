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
        private static void RegisterReduceWindow2D(CodeGenRegistry registry)
        {
            registry.Add<ReduceWindow2D>((n, g) =>
            {
                return new ReduceWindow2DOptionsBody
                {
                    Options = new ReduceWindow2DOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        InputShape = OpUtility.To(n.Input.Shape),
                        ReduceOperator = n.ReduceOperator,
                        PaddingH = n.PaddingH,
                        PaddingW = n.PaddingW,
                        FilterH = n.FilterH,
                        FilterW = n.FilterW,
                        StrideH = n.StrideH,
                        StrideW = n.StrideW,
                        DilationH = n.DilationH,
                        DilationW = n.DilationW,
                        InitialValue = n.InitialValue,
                        FusedActivation = n.FusedActivation
                    }
                };
            });
        }
    }
}
