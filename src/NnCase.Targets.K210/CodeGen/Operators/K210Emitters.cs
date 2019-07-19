using System;
using System.Collections.Generic;
using System.Text;
using NnCase.CodeGen;
using NnCase.Evaluation;

namespace NnCase.Targets.K210.CodeGen.Operators
{
    internal static partial class K210Emitters
    {
        public static void Register(CodeGenRegistry registry)
        {
            RegisterKPUUpload(registry);
            RegisterKPUConv2D(registry);
        }
    }
}
