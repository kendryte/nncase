// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.CPU;

public static class KernelUtility
{
    public static string DimensionsToC(Shape dimensions)
    {
        var sb = new StringBuilder("fixed_shape<");
        for (int i = 0; i < dimensions.Count; i++)
        {
            var value = dimensions[i].FixedValue;
            sb.Append(value);
            if (i != dimensions.Count - 1)
            {
                sb.Append(", ");
            }
        }

        sb.Append('>');
        return sb.ToString();
    }

    public static string DimensionsToC(ReadOnlySpan<Expr> dimensions)
    {
        var sb = new StringBuilder("fixed_shape<");
        for (int i = 0; i < dimensions.Length; i++)
        {
            var value = ((TensorConst)dimensions[i]).Value.Cast<ulong>()[0];
            sb.Append(value);
            if (i != dimensions.Length - 1)
            {
                sb.Append(", ");
            }
        }

        sb.Append('>');
        return sb.ToString();
    }

    public static string StridesToC(ReadOnlySpan<Expr> dimensions)
    {
        var sb = new StringBuilder("fixed_strides<");
        for (int i = 0; i < dimensions.Length; i++)
        {
            var value = ((TensorConst)dimensions[i]).Value.Cast<ulong>()[0];
            sb.Append(value);
            if (i != dimensions.Length - 1)
            {
                sb.Append(", ");
            }
        }

        sb.Append('>');
        return sb.ToString();
    }

    public static string NdSBPToC(IRArray<SBP> ndSBP)
    {
        var sb = new StringBuilder("dist<");
        for (int i = 0; i < ndSBP.Count; i++)
        {
            var value = ndSBP[i];
            if (value is SBPBroadCast)
            {
                sb.Append('B');
            }
            else if (value is SBPPartialSum)
            {
                sb.Append('P');
            }
            else if (value is SBPSplit split)
            {
                sb.Append($"S<{split.Axis}>");
            }

            if (i != ndSBP.Count - 1)
            {
                sb.Append(", ");
            }
        }

        sb.Append('>');
        return sb.ToString();
    }
}
