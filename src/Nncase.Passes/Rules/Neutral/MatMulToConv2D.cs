// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
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
        IsMatMul(
            "matMul",
            "matMulCall",
            _ => true,
            IsWildcard("a") with { TypePattern = HasRank(2) & HasFixedShape() },
            IsTensorConst("b") with { TypePattern = HasRank(2) & HasFixedShape() });

    private Expr? GetReplace(Call matMulCall, Expr a, Expr b)
    {
        var aShape = a.CheckedShape;
        var bShape = b.CheckedShape;
        if (aShape[1] != bShape[0])
        {
            return null;
        }

        var if_shape = new Shape(new[] { aShape[0].FixedValue, aShape[1].FixedValue, 1, 1 });
        var w_shape = new Shape(new[] { bShape[1].FixedValue, bShape[0].FixedValue, 1, 1 });
        var of_shape = new Shape(new[] { aShape[0].FixedValue, bShape[1].FixedValue });

        var if_reshape = Reshape(a, if_shape);
        var w_tp = Transpose(b, Tensor.From<int>(new[] { 1, 0 })).InheritMetaData(b);
        var w_reshape = Reshape(w_tp, w_shape).InheritMetaData(b);
        var conv2d = Conv2D(
            if_reshape,
            w_reshape,
            Tensor.FromScalar(0.0f, w_shape[0].FixedValue),
            Tensor.FromScalar(1, new[] { 2 }),
            Tensor.FromScalar(0, new[] { 2, 2 }),
            new int[] { 1, 1 },
            PadMode.Constant,
            1).InheritMetaData(matMulCall);
        return Reshape(conv2d, of_shape).InheritMetaData(matMulCall);
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
        IsMatMul(
            "matMul",
            "matMulCall",
            _ => true,
            IsWildcard("a") with { TypePattern = HasRank(3) & HasFixedShape() },
            IsTensorConst("b") with { TypePattern = HasRank(2) & HasFixedShape() });

    private Expr? GetReplace(Call matMulCall, Expr a, Expr b)
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
            if_reshape,
            w_reshape,
            Tensor.FromScalar(0.0f, w_shape[0].FixedValue),
            Tensor.FromScalar(1, new[] { 2 }),
            Tensor.FromScalar(0, new[] { 2, 2 }),
            new int[] { 1, 1 },
            PadMode.Constant,
            1).InheritMetaData(matMulCall);
        return Reshape(conv2d, of_shape).InheritMetaData(matMulCall);
    }
}

/// <summary>
/// Transform broadcast <see cref="IR.Math.MatMul"/> b to <see cref="IR.Math.MatMul"/> a and squeeze matmul to 3D.
/// </summary>
[RuleGenerator]
public sealed partial class BroadcastMatMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsMatMul(
            "matMul",
            "matMulCall",
            _ => true,
            IsWildcard("a") with { TypePattern = HasRank(r => r > 2, "Rank > 2") & HasFixedShape() },
            IsWildcard("b") with { TypePattern = HasRank(r => r > 2, "Rank > 2") & HasFixedShape() });

    private Expr? GetReplace(Call matMulCall, Expr a, Expr b)
    {
        var aShape = a.CheckedShape;
        var bShape = b.CheckedShape;

        var sizeA = aShape.Size;
        var sizeB = bShape.Size;

        if (sizeA / (aShape[^1].FixedValue * aShape[^2].FixedValue) == sizeB / (bShape[^1].FixedValue * bShape[^2].FixedValue) && aShape.Rank == 3 && bShape.Rank == 3)
        {
            return null;
        }

        if (aShape.Rank > bShape.Rank)
        {
            var newBShape = aShape.ToValueArray();
            newBShape[^1] = bShape[^1].FixedValue;
            newBShape[^2] = bShape[^2].FixedValue;

            var newOutputShape = aShape.ToValueArray();
            newOutputShape[^2] = aShape[^2].FixedValue;
            newOutputShape[^1] = bShape[^1].FixedValue;

            var ifShape = new int[] { -1, aShape[^2].FixedValue, aShape[^1].FixedValue };
            var wShape = new int[] { -1, newBShape[^2], newBShape[^1] };
            var bBroadCast = IR.F.Tensors.Broadcast(b, newBShape);
            List<string> outputNames = new() { b.Metadata.OutputNames![0] + "_bBroadCast" };
            bBroadCast.Metadata.OutputNames = outputNames;
            return Reshape(MatMul(Reshape(a, ifShape), Reshape(bBroadCast, wShape)), newOutputShape);
        }
        else if (aShape.Rank < bShape.Rank)
        {
            var newAShape = bShape.ToValueArray();
            newAShape[^1] = aShape[^1].FixedValue;
            newAShape[^2] = aShape[^2].FixedValue;

            var newOutputShape = bShape.ToValueArray();
            newOutputShape[^2] = aShape[^2].FixedValue;
            newOutputShape[^1] = bShape[^1].FixedValue;

            var ifShape = new int[] { -1, newAShape[^2], newAShape[^1] };
            var wShape = new int[] { -1, bShape[^2].FixedValue, bShape[^1].FixedValue };
            var aBroadCast = IR.F.Tensors.Broadcast(a, newAShape);
            List<string> outputNames = new() { a.Metadata.OutputNames![0] + "_aBroadCast" };
            aBroadCast.Metadata.OutputNames = outputNames;
            return Reshape(MatMul(Reshape(aBroadCast, ifShape), Reshape(b, wShape)), newOutputShape);
        }
        else
        {
            var newAShape = aShape.ToValueArray();
            var newBShape = bShape.ToValueArray();
            var newOutputShape = aShape.ToValueArray();
            newOutputShape[^2] = aShape[^2].FixedValue;
            newOutputShape[^1] = bShape[^1].FixedValue;

            for (int i = 0; i < aShape.Rank - 2; i++)
            {
                newAShape[i] = aShape[i].FixedValue == 1 ? bShape[i].FixedValue : aShape[i].FixedValue;
                newBShape[i] = bShape[i].FixedValue == 1 ? aShape[i].FixedValue : bShape[i].FixedValue;
                newOutputShape[i] = System.Math.Max(aShape[i].FixedValue, bShape[i].FixedValue);
            }

            var ifShape = new int[] { -1, newAShape[^2], newAShape[^1] };
            var wShape = new int[] { -1, newBShape[^2], newBShape[^1] };
            var bBroadCast = IR.F.Tensors.Broadcast(b, newBShape);
            List<string> bOutputNames = new() { b.Metadata.OutputNames?[0] + "_bBroadCast" };
            bBroadCast.Metadata.OutputNames = bOutputNames;
            var aBroadCast = IR.F.Tensors.Broadcast(a, newAShape);
            List<string> aOutputNames = new() { a.Metadata.OutputNames?[0] + "_aBroadCast" };
            aBroadCast.Metadata.OutputNames = aOutputNames;
            return Reshape(
                        MatMul(
                            Reshape(
                                aBroadCast,
                                ifShape),
                            Reshape(
                                bBroadCast,
                                wShape)).InheritMetaData(matMulCall),
                        newOutputShape);
        }
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
