using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;

namespace NnCase.Targets.K210.Evaluation.Operators
{
    internal static partial class K210Evaulators
    {
        public static void Register(EvaluatorRegistry registry)
        {
            RegisterKPUFakeConv2D(registry);
        }
    }
}
