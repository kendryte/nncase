using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.CodeGen;
using NnCase.Evaluation;
using NnCase.Evaluation.Operators;
using NnCase.Runtime.Operators;
using NnCase.Targets.CPU.IR.Operators;

namespace NnCase.Targets.CPU.CodeGen.Operators
{
    internal static partial class CPUEmitters
    {
        private static void RegisterCPUReduceWindow2D(CodeGenRegistry registry)
        {
            registry.Add<CPUReduceWindow2D>((n, g) =>
            {
                return new CPUReduceWindow2DOptionsBody
                {
                    Options = new CPUReduceWindow2DOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        ReduceOperator = n.ReduceOperator,
                        InputShape = OpUtility.To(n.Input.Shape),
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
