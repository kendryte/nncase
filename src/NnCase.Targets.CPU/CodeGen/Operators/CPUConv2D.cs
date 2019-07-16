using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.CodeGen;
using NnCase.Evaluation;
using NnCase.Evaluation.Operators;
using NnCase.Targets.CPU.IR.Operators;
using NnCase.Targets.CPU.Runtime.Operators;

namespace NnCase.Targets.CPU.CodeGen.Operators
{
    internal static partial class CPUEmitters
    {
        private static void RegisterCPUConv2D(CodeGenRegistry registry)
        {
            registry.Add<CPUConv2D>((n, g) =>
            {
                return new CPUConv2DOptionsBody
                {
                    Options = new CPUConv2DOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        InputShape = OpUtility.To(n.Input.Shape),
                        OutputChannels = n.Output.Shape[3],
                        PaddingH = n.PaddingH,
                        PaddingW = n.PaddingW,
                        FilterH = n.Weights.Dimensions[1],
                        FilterW = n.Weights.Dimensions[2],
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
