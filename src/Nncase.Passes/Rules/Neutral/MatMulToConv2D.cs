// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;
using Shape = Nncase.IR.Shape;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Transform <see cref="IR.Math.MatMul"/> to <see cref="IR.NN.Conv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class MatMulToConv2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsRangeOfMarker("marker", IsMatMul(
            "matMul",
            "matMulCall",
            _ => true,
            IsRangeOfMarker("am", IsWildcard("a") with { TypePattern = HasFixedShape() }, IsWildcard()),
            IsRangeOfMarker("bm", IsTensorConst("b") with { TypePattern = HasRank(2) & HasFixedShape() }, IsWildcard())),
            IsWildcard());

    private static int counter = 0;
    private Expr? GetReplace(Marker marker, Call matMulCall, Expr a, Expr b, Marker am, Marker bm)
    {
        var aShape = a.CheckedShape;
        var bShape = b.CheckedShape;
        if (aShape.Count > 2 && aShape.ToValueArray()[..^2].Aggregate(1, (sum, x) => sum * x) != 1)
        {
            return null;
        }
        if (aShape[^1] != bShape[^2])
        {
            return null;
        }

        var if_shape = new Shape(new[] { aShape[^2].FixedValue, aShape[^1].FixedValue, 1, 1 });
        var w_shape = new Shape(new[] { bShape[^1].FixedValue, bShape[^2].FixedValue, 1, 1 });
        var of_shape = new Shape(new[] { aShape[^2].FixedValue, bShape[^1].FixedValue });

        var if_reshape = Reshape(a, if_shape);
        var w_tp = Transpose(b, Tensor.From<int>(new[] { 1, 0 })).InheritMetaData(b);
        var w_reshape = Reshape(w_tp, w_shape).InheritMetaData(b);
        var conv2d = Conv2D(
            am.With(target: if_reshape),
            bm.With(target: w_reshape),
            Tensor.FromScalar(0.0f, w_shape[0].FixedValue),
            Tensor.FromScalar(1, new[] { 2 }),
            Tensor.FromScalar(0, new[] { 2, 2 }),
            new int[] { 1, 1 },
            PadMode.Constant,
            1).InheritMetaData(matMulCall);
        var m = Reshape(marker.With(target: conv2d), of_shape).InheritMetaData(matMulCall);
        DumpScope.Current.DumpIR(m, $"{counter++}", "withMarker");
        return m;
    }
}

/// <summary>
/// Transform broadcast <see cref="IR.Math.MatMul"/> to <see cref="IR.NN.Conv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class BroadcastMatMulToConv2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsRangeOfMarker("marker", IsMatMul(
                "matMul",
                "matMulCall",
                _ => true,
                IsRangeOfMarker("am", IsWildcard("a") with { TypePattern = HasRank(3) & HasFixedShape() }, IsWildcard()),
                IsRangeOfMarker("bm", IsTensorConst("b") with { TypePattern = HasRank(2) & HasFixedShape() }, IsWildcard())),
            IsWildcard());

    private Expr? GetReplace(Marker marker, Call matMulCall, Expr a, Expr b, Marker am, Marker bm)
    {
        var aShape = a.CheckedShape;
        var bShape = b.CheckedShape;
        if (aShape[2] != bShape[0])
        {
            return null;
        }

        var if_shape = new Shape(new[] { aShape[0].FixedValue * aShape[1].FixedValue, aShape[2].FixedValue, 1, 1 });
        var w_shape = new Shape(new[] { bShape[1].FixedValue, bShape[0].FixedValue, 1, 1 });
        var of_shape = new Shape(new[] { aShape[0].FixedValue, aShape[1].FixedValue, bShape[1].FixedValue });

        var if_reshape = Reshape(a, if_shape);
        var w_tp = Transpose(b, Tensor.From<int>(new[] { 1, 0 })).InheritMetaData(b);
        var w_reshape = Reshape(w_tp, w_shape).InheritMetaData(b);

        var conv2d = Conv2D(
            am.With(target: if_reshape),
            bm.With(target: w_reshape),
            Tensor.FromScalar(0.0f, w_shape[0].FixedValue),
            Tensor.FromScalar(1, new[] { 2 }),
            Tensor.FromScalar(0, new[] { 2, 2 }),
            new int[] { 1, 1 },
            PadMode.Constant,
            1).InheritMetaData(matMulCall);
        var m = Reshape(marker.With(target: conv2d), of_shape).InheritMetaData(matMulCall);
        return m;
        // return Reshape(conv2d, of_shape).InheritMetaData(matMulCall);
    }
}

/// <summary>
/// Transform non-broadcast multiple <see cref="IR.Math.MatMul"/>.
/// </summary>
[RuleGenerator]
public sealed partial class SplitBatchMatMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsMatMul(
            "matMul",
            "matMulCall",
            _ => true,
            IsWildcard("a") with { TypePattern = HasRank(3) & HasFixedShape() },
            IsTensorConst("b") with { TypePattern = HasRank(3) & HasFixedShape() });

    private Expr? GetReplace(Call matMulCall, Expr a, Expr b)
    {
        var aShape = a.CheckedShape;
        var bShape = b.CheckedShape;
        if (aShape[2] != bShape[1] || aShape[0] != bShape[0])
        {
            return null;
        }

        var ifSlices = new Expr[aShape[0].FixedValue];
        var wSlices = new Expr[aShape[0].FixedValue];
        var mmSlices = new Expr[aShape[0].FixedValue];
        var ofSlices = new Expr[aShape[0].FixedValue];

        var if_shape = new Shape(new[] { aShape[1].FixedValue, aShape[2].FixedValue });
        var w_shape = new Shape(new[] { bShape[1].FixedValue, bShape[2].FixedValue });

        for (var i = 0; i < aShape[0].FixedValue; i++)
        {
            var begin = new[] { i };
            var ifEnd = new[] { i + 1 };
            var wEnd = new[] { i + 1 };
            ifSlices[i] = Reshape(Slice(a, begin, ifEnd, new[] { 0 }, new[] { 1 }), if_shape);
            wSlices[i] = Reshape(Slice(b, begin, wEnd, new[] { 0 }, new[] { 1 }), w_shape);
            mmSlices[i] = MatMul(ifSlices[i], wSlices[i]);
            ofSlices[i] = Reshape(mmSlices[i], new Shape(1, aShape[1].FixedValue, bShape[2].FixedValue));
        }

        return Concat(new IR.Tuple(ofSlices), 0).InheritMetaData(matMulCall);
    }
}
