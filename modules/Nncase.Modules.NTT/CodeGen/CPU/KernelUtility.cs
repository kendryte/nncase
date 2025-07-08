// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.CodeGen.NTT;

public static class KernelUtility
{
    public static string DimensionsToC(bool isFixed, ReadOnlySpan<CSymbol> dimensions, bool isType) =>
        DimensionsToC("shape", isFixed, dimensions, isType);

    public static string StridesToC(bool isFixed, ReadOnlySpan<CSymbol> dimensions, bool isType) =>
        DimensionsToC("strides", isFixed, dimensions, isType);

    public static string DimensionsTypeToC(bool isFixed, ReadOnlySpan<Dimension> dimensions) =>
        DimensionsTypeToC("shape", isFixed, dimensions);

    public static string StridesTypeToC(bool isFixed, ReadOnlySpan<Dimension> dimensions) =>
        DimensionsTypeToC("strides", isFixed, dimensions);

    public static string PlacementToC(this Placement placement)
    {
        return $"mesh<topology::thread, {string.Join(',', placement.Hierarchy)}>";
    }

    public static string ShardingToC(DistributedType distributedType)
    {
        var placement = distributedType.Placement;
        var ndSBP = distributedType.AxisPolicies;

        var sb = new StringBuilder("make_sharding<mesh<topology::thread, ");
        for (int i = 0; i < placement.Rank; i++)
        {
            var value = placement.Hierarchy[i];
            sb.Append($"{value}");
            if (i != placement.Rank - 1)
            {
                sb.Append(", ");
            }
        }

        sb.Append(">>(");
        for (int axis = 0; axis < distributedType.TensorType.Shape.Rank; axis++)
        {
            var value = ndSBP[axis];
            if (value is SBPSplit s)
            {
                sb.Append($"S<{string.Join(", ", s.Axes)}>()");
            }
            else
            {
                sb.Append('B');
            }

            if (axis != distributedType.TensorType.Shape.Rank - 1)
            {
                sb.Append(", ");
            }
        }

        sb.Append(')');
        return sb.ToString();
    }

    private static string DimensionsToC(string typeName, bool isFixed, ReadOnlySpan<CSymbol> dimensions, bool isType)
    {
        if (isFixed)
        {
            var sb = new StringBuilder($"fixed_{typeName}<");
            AppendDimValues(sb, dimensions);
            sb.Append(isType ? ">" : ">{}");
            return sb.ToString();
        }
        else
        {
            if (isType)
            {
                return $"ranked_{typeName}<{dimensions.Length}>";
            }
            else
            {
                var sb = new StringBuilder($"make_ranked_{typeName}(");
                AppendDimValues(sb, dimensions);
                sb.Append(')');
                return sb.ToString();
            }
        }
    }

    private static string DimensionsTypeToC(string typeName, bool isFixed, ReadOnlySpan<Dimension> dimensions)
    {
        if (isFixed)
        {
            var sb = new StringBuilder($"fixed_{typeName}<");
            AppendDimValues(sb, dimensions);
            sb.Append('>');
            return sb.ToString();
        }
        else
        {
            return $"ranked_{typeName}<{dimensions.Length}>";
        }
    }

    private static void AppendDimValues(StringBuilder sb, ReadOnlySpan<CSymbol> dimensions)
    {
        for (int i = 0; i < dimensions.Length; i++)
        {
            var value = dimensions[i].Name;
            sb.Append(value);
            if (i != dimensions.Length - 1)
            {
                sb.Append(", ");
            }
        }
    }

    private static void AppendDimValues(StringBuilder sb, ReadOnlySpan<Dimension> dimensions)
    {
        for (int i = 0; i < dimensions.Length; i++)
        {
            var value = dimensions[i].FixedValue;
            sb.Append(value);
            if (i != dimensions.Length - 1)
            {
                sb.Append(", ");
            }
        }
    }
}
