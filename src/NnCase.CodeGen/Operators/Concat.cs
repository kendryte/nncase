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
        private static void RegisterConcat(CodeGenRegistry registry)
        {
            registry.Add<Concat>((n, g) =>
            {
                (var innerSize, var outerSize) = OpUtility.GetConcatParams(n.Output.Shape, ShapeUtility.GetBytes(n.Output.Type), n.Axis);

                return new ConcatOptionsBody
                {
                    Options = new ConcatOptions
                    {
                        Output = g.MemoryRange(n.Output),
                        InnerSize = innerSize,
                        OuterSize = outerSize,
                        InputsCount = n.Inputs.Count,
                        Inputs = n.Inputs.Select(x => g.MemoryRange(x)).ToArray(),
                        Dimensions = n.Inputs.Select(x => x.Shape[n.Axis]).ToArray()
                    }
                };
            });
        }
    }
}
