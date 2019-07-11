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
        private static void RegisterDequantize(CodeGenRegistry registry)
        {
            registry.Add<Dequantize>((n, g) =>
            {
                return new DequantizeOptionsBody
                {
                    Options = new DequantizeOptions
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
