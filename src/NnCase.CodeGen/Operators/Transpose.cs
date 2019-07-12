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
        private static void RegisterTranspose(CodeGenRegistry registry)
        {
            registry.Add<Transpose>((n, g) =>
            {
                var elementSize = ShapeUtility.GetBytes(n.Input.Type);
                (var rtInShape, var rtPerm) = OpUtility.ExtendTransposeShape(n.Input.Shape, n.Perm);

                return new TransposeOptionsBody
                {
                    Options = new TransposeOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        InputShape = rtInShape,
                        Perm = rtPerm,
                        ElementSize = elementSize
                    }
                };
            });
        }
    }
}
