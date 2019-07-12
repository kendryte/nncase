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
        private static void RegisterReshape(CodeGenRegistry registry)
        {
            registry.Add<Reshape>((n, g) =>
            {
                return new MemoryCopyOptionsBody
                {
                    Options = new MemoryCopyOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output)
                    }
                };
            });
        }
    }
}
