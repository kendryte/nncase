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
        private static void RegisterPad(CodeGenRegistry registry)
        {
            registry.Add<Pad>((n, g) =>
            {
                return new PadOptionsBody
                {
                    Options = new PadOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        Paddings = OpUtility.To(n.Paddings),
                        PadValue = n.PadValue
                    }
                };
            });
        }
    }
}
