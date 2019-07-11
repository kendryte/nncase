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
        private static void RegisterConv2D(CodeGenRegistry registry)
        {
            registry.Add<Conv2D>((n, g) =>
            {
                return new Conv2DOptionsBody
                {
                    Options = new Conv2DOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        InputShape = OpUtility.To(n.Input.Shape),
                        Groups = n.Groups,
                        OutputChannels = n.Output.Shape[1],
                        PaddingH = n.PaddingH,
                        PaddingW = n.PaddingW,
                        FilterH = n.Weights.Dimensions[2],
                        FilterW = n.Weights.Dimensions[3],
                        StrideH = n.StrideH,
                        StrideW = n.StrideW,
                        DilationH = n.DilationH,
                        DilationW = n.DilationW,
                        FusedActivation = n.FusedActivation,
                        Weights = n.Weights.Buffer.ToArray(),
                        Bias = n.Bias.Buffer.ToArray()
                    }
                };
            });
        }
    }
}
