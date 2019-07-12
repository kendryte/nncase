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
        private static void RegisterQuantize(CodeGenRegistry registry)
        {
            registry.Add<Quantize>((n, g) =>
            {
                return new QuantizeOptionsBody
                {
                    Options = new QuantizeOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        QuantizationParam = n.QuantizationParam
                    }
                };
            });
        }
    }
}
