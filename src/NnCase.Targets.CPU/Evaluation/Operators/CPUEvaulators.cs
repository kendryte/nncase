using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;

namespace NnCase.Targets.CPU.Evaluation.Operators
{
    internal static partial class CPUEvaulators
    {
        public static void Register(EvaluatorRegistry registry)
        {
            RegisterCPUConv2D(registry);
            RegisterCPUDepthwiseConv2D(registry);
            RegisterCPUReduceWindow2D(registry);
        }
    }
}
