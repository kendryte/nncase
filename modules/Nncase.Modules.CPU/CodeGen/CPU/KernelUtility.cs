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

    public static string DistributedToC(DistributedType distributedType)
    {
        var placement = distributedType.Placement;
        var ndSBP = distributedType.NdSBP;

        var sb = new StringBuilder("sharding<mesh<topology::thread, ");
        for (int i = 0; i < placement.Rank; i++)
        {
            var value = placement.Hierarchy[i];
            sb.Append($"{value}");
            if (i != placement.Rank - 1)
            {
                sb.Append(", ");
            }
        }

        var implicitPolicy = ndSBP.Any(x => x is SBPPartialSum) ? "P<reduce_op::sum>" : "B";
        sb.Append($">, {implicitPolicy}");

        for (int axis = 0; axis < distributedType.TensorType.Shape.Rank; axis++)
        {
            var value = from sbp in ndSBP.Select((x, i) => (x, i))
                        where sbp.x is SBPSplit split && split.Axis == axis
                        select sbp.i;
            if (value.Any())
            {
                sb.Append($", S<{string.Join(", ", value)}>");
            }
            else
            {
                sb.Append($", I");
            }
        }

        sb.Append('>');
        return sb.ToString();
    }
}
