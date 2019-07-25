using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;

namespace NnCase.CodeGen.Operators
{
    internal static partial class DefaultEmitters
    {
        public static void Register(CodeGenRegistry registry)
        {
            RegisterPlaceholders(registry);
            RegisterBinary(registry);
            RegisterConcat(registry);
            RegisterConv2D(registry);
            RegisterDequantize(registry);
            RegisterMatMul(registry);
            RegisterPad(registry);
            RegisterQuantize(registry);
            RegisterReduce(registry);
            RegisterReduceWindow2D(registry);
            RegisterReshape(registry);
            RegisterSoftmax(registry);
            RegisterTranspose(registry);
            RegisterStridedSlice(registry);
            RegisterResizeBilinear(registry);
            RegisterResizeNearestNeighbor(registry);
        }
    }
}
