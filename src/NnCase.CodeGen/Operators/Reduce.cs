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
        private static void RegisterReduce(CodeGenRegistry registry)
        {
            registry.Add<Reduce>((n, g) =>
            {
                var reducedShape = ShapeUtility.GetReducedShape(n.Input.Shape, n.Axis, true);

                return new ReduceOptionsBody
                {
                    Options = new ReduceOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        ReduceOperator = n.ReduceOperator,
                        InputShape = OpUtility.To(n.Input.Shape),
                        OutputShape = OpUtility.To(reducedShape),
                        InitialValue = n.InitialValue
                    }
                };
            });
        }
    }
}
