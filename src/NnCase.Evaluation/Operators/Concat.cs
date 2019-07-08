using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Kernels;

namespace NnCase.Evaluation.Operators
{
    internal static partial class DefaultEvaulators
    {
        private static void RegisterConcat(EvaluatorRegistry registry)
        {
            registry.Add<Concat>((n, e) =>
            {
                var inputs = n.Inputs.Select(x => (ReadOnlyMemory<byte>)e.MemoryAt(x)).ToList();
                var output = e.MemoryAt<byte>(n.Output);

                (var innerSize, var outerSize) = OpUtility.GetConcatParams(n.Output.Shape, ShapeUtility.GetBytes(n.Output.Type), n.Axis);
                DefaultKernels.Concat(inputs, output, n.Inputs.Select(x => x.Shape[n.Axis]).ToList(), innerSize, outerSize);
            });
        }
    }
}
