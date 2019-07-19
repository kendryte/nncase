using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.CodeGen;
using NnCase.Evaluation;
using NnCase.Evaluation.Operators;
using NnCase.Targets.K210.IR.FakeOperators;
using NnCase.Targets.K210.Runtime.Operators;

namespace NnCase.Targets.K210.CodeGen.Operators
{
    internal static partial class K210Emitters
    {
        private static void RegisterKPUUpload(CodeGenRegistry registry)
        {
            registry.Add<KPUUpload>((n, g) =>
            {
                return new KPUUploadOptionsBody
                {
                    Options = new KPUUploadOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        InputShape = OpUtility.To(n.Input.Shape)
                    }
                };
            });
        }
    }
}
