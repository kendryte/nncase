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
        private static void RegisterResizeNearestNeighbor(CodeGenRegistry registry)
        {
            registry.Add<ResizeNearestNeighbor>((n, g) =>
            {
                return new ResizeNearestNeighborOptionsBody
                {
                    Options = new ResizeNearestNeighborOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        InputShape = OpUtility.To(n.Input.Shape),
                        OutputHeight = n.OutputHeight,
                        OutputWidth = n.OutputWidth,
                        AlignCorners = n.AlignCorners
                    }
                };
            });
        }
    }
}
