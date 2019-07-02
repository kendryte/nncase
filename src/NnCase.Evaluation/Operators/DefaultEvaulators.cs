using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Evaluation.Operators
{
    internal static partial class DefaultEvaulators
    {
        static partial void RegisterPartial(EvaluatorRegistry registry);

        public static void Register(EvaluatorRegistry registry)
        {
            RegisterPartial(registry);
        }
    }
}
