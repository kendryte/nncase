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
        private static void RegisterSoftmax(CodeGenRegistry registry)
        {
            registry.Add<Softmax>((n, g) =>
            {
                return new SoftmaxOptionsBody
                {
                    Options = new SoftmaxOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        InnerSize = ShapeUtility.ComputeSize(n.Input.Shape) / n.Input.Shape[0],
                        OuterSize = n.Input.Shape[0],
                        Beta = n.Beta
                    }
                };
            });
        }
    }
}
