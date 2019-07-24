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
        private static void RegisterStridedSlice(CodeGenRegistry registry)
        {
            registry.Add<StridedSlice>((n, g) =>
            {
                return new StridedSliceOptionsBody
                {
                    Options = new StridedSliceOptions
                    {
                        Input = g.MemoryRange(n.Input),
                        Output = g.MemoryRange(n.Output),
                        InputShape = OpUtility.To(n.Input.Shape),
                        Begin = OpUtility.To(n.Begin),
                        End = OpUtility.To(n.End),
                        Strides = OpUtility.To(n.Strides),
                        BeginMask = n.BeginMask,
                        EndMask = n.EndMask,
                        EllipsisMask = n.EllipsisMask,
                        NewAxisMask = n.NewAxisMask,
                        ShrinkAxisMask = n.ShrinkAxisMask
                    }
                };
            });
        }
    }
}
