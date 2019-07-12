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
        private static void RegisterResizeBilinear(CodeGenRegistry registry)
        {
            registry.Add<ResizeBilinear>((n, g) =>
            {
                return new ResizeBilinearOptionsBody
                {
                    Options = new ResizeBilinearOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        OutputHeight = n.OutputHeight,
                        OutputWidth = n.OutputWidth,
                        AlignCorners = n.AlignCorners
                    }
                };
            });
        }
    }
}
