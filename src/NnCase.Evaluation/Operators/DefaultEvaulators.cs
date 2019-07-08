using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Evaluation.Operators
{
    internal static partial class DefaultEvaulators
    {
        public static void Register(EvaluatorRegistry registry)
        {
            RegisterPlaceholders(registry);
            RegisterTranspose(registry);
            RegisterConv2D(registry);
            RegisterReduceWindow2D(registry);
            RegisterReshape(registry);
            RegisterBinary(registry);
            RegisterConcat(registry);
        }
    }
}
