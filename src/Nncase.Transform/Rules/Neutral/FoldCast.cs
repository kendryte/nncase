// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Fold two <see cref="IR.Tensors.Cast"/> into one <see cref="IR.Tensors.Cast"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldTwoCasts : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsCast("c2", x => true, IsCast("c1", x => true, IsWildcard("x")));

    private Expr? GetReplace(Expr x, Cast c1, Cast c2)
    {
        if (IsLosslessCast(x.CheckedDataType, c1.NewType))
        {
            return Cast(x, c2.NewType);
        }

        return null;
    }

    private bool IsLosslessCast(DataType pre, DataType post)
    {
        // TODO: Support more types
        if (pre == DataTypes.UInt8)
        {
            return post == DataTypes.UInt8
                || post == DataTypes.UInt16
                || post == DataTypes.UInt32
                || post == DataTypes.UInt64
                || post == DataTypes.Int16
                || post == DataTypes.Int32
                || post == DataTypes.Int64
                || post == DataTypes.Float16
                || post == DataTypes.Float32
                || post == DataTypes.Float64
                || post == DataTypes.BFloat16;
        }
        else if (pre == DataTypes.Int8)
        {
            return post == DataTypes.UInt16
                || post == DataTypes.UInt32
                || post == DataTypes.UInt64
                || post == DataTypes.Int16
                || post == DataTypes.Int32
                || post == DataTypes.Int64
                || post == DataTypes.Float16
                || post == DataTypes.Float32
                || post == DataTypes.Float64
                || post == DataTypes.BFloat16;
        }
        else if (pre == DataTypes.UInt16)
        {
            return post == DataTypes.UInt16
                || post == DataTypes.UInt32
                || post == DataTypes.UInt64
                || post == DataTypes.Int32
                || post == DataTypes.Int64
                || post == DataTypes.Float32
                || post == DataTypes.Float64;
        }

        return false;
    }
}

/// <summary>
/// Fold nop <see cref="IR.Tensors.Cast"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopCast : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsCast("c1", x => true, IsWildcard("x"));

    private Expr? GetReplace(Expr x, Cast c1)
    {
        if (c1.NewType == x.CheckedDataType)
        {
            return x;
        }

        return null;
    }
}
