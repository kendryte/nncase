using System;
using System.Collections.Generic;
using System.Text;
using NnCase.CodeGen;
using NnCase.Evaluation;

namespace NnCase.Targets.CPU.CodeGen.Operators
{
    internal static partial class CPUEmitters
    {
        public static void Register(CodeGenRegistry registry)
        {
            RegisterCPUConv2D(registry);
            RegisterCPUDepthwiseConv2D(registry);
            RegisterCPUReduceWindow2D(registry);
        }
    }
}
