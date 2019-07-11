using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.CodeGen.Operators
{
    internal static partial class DefaultEmitters
    {
        private static void RegisterPlaceholders(CodeGenRegistry registry)
        {
            registry.DisableRuntime<InputNode>();
            registry.DisableRuntime<OutputNode>();
            registry.DisableRuntime<Constant>();
        }
    }
}
