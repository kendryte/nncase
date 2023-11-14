// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.PatternMatch;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Conv2D = Nncase.IR.NN.Conv2D;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class FoldDilatedConv2D : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
        IsBatchToSpace(
            "bts",
            "btsCall",
            IsRangeOfMarker(
                    "btsInput",
                    Conv2DPattern(),
                    IsTensorConst()),
            IsTensorConst("btsBlockShape"),
            IsTensorConst("originCrop"));

    private static CallPattern Conv2DPattern() =>
        IsCallWildcard(
            "conv",
            IsOp<Conv2D>(),
            IsRangeOfMarker(
                    IsSpaceToBatch(
                        "sbt",
                        "stbCall",
                        IsWildcard("stbInput") with { TypePattern = HasFixedShape() },
                        IsTensorConst("stbBlockShape"),
                        IsTensorConst("originPaddings")),
                    IsTensorConst()));

    private Expr? GetReplace(Call conv, Call btsCall, Call stbCall, Expr btsInput, Expr stbInput, int[] btsBlockShape, int[] stbBlockShape, int[] originPaddings, int[] originCrop)
    {
        var btsShape = btsCall.CheckedShape.ToValueArray();
        var btsInputShape = btsInput.CheckedShape.ToValueArray();
        var stbInputShape = stbInput.CheckedShape.ToValueArray();

        var paddings = new[,] { { originPaddings[0], originPaddings[1] }, { originPaddings[2], originPaddings[3] } };
        var crop = new[,] { { originCrop[0], originCrop[1] }, { originCrop[2], originCrop[3] } };

        var padIfH = paddings[0, 0] + paddings[0, 1] + stbInputShape[2];
        var padIfW = paddings[1, 0] + paddings[1, 1] + stbInputShape[3];
        var dilationH = stbBlockShape[0];
        var dilationW = stbBlockShape[1];
        var weightsShape = conv.Arguments[Conv2D.Weights.Index].CheckedShape.ToValueArray();
        var wH = weightsShape[2];
        var wW = weightsShape[3];
        var outH = btsShape[2] + crop[0, 0] + crop[0, 1];
        var outW = btsShape[3] + crop[1, 0] + crop[1, 1];
        var strideH = outH == 1 ? 1 : (padIfH - (dilationH * (wH - 1)) - 1) / (outH - 1);
        var strideW = outW == 1 ? 1 : (padIfW - (dilationW * (wW - 1)) - 1) / (outW - 1);

        var (begin, end) = GetBeginEnd(btsBlockShape, crop, btsInputShape);
        var slicePadding = new[,]
        {
            { -begin[0], end[0] - btsShape[0] },
            { -begin[3], end[3] - btsShape[1] },
            { -begin[1], end[1] - btsShape[2] },
            { -begin[2], end[2] - btsShape[3] },
        };

        var newPaddings = new[,]
        {
            { 0, 0 },
            { 0, 0 },
            { paddings[0, 0] + (strideH * slicePadding[2, 0]) - crop[0, 0], paddings[0, 1] + (strideH * slicePadding[2, 1]) - crop[0, 1] },
            { paddings[1, 0] + (strideH * slicePadding[3, 0]) - crop[1, 0], paddings[1, 1] + (strideH * slicePadding[3, 1]) - crop[1, 1] },
        };

        var pairs = new[]
        {
            (Conv2D.Input.Index, stbInput),
            (Conv2D.Padding.Index, (Expr)newPaddings),
            (Conv2D.Stride.Index, (Expr)new[] { strideH, strideW }),
            (Conv2D.Dilation.Index, (Expr)new[] { dilationH, dilationW }),
        };
        return ReplaceUtility.ReplaceCallParams(conv, conv.Arguments.ToArray(), pairs);
    }

    private (int[] Begin, int[] End) GetBeginEnd(int[] btsBlockShape, int[,] crop, int[] btsInputShape)
    {
        List<int> shape_expend = new();
        var block_shape_produt = btsBlockShape.Aggregate((x, sum) => x * sum);
        for (var i = 0; i < btsBlockShape.Length; i++)
        {
            shape_expend.Add(btsBlockShape[i]);
        }

        shape_expend.Add(btsInputShape[0] / block_shape_produt);
        for (var i = 1; i < btsInputShape.Length; i++)
        {
            shape_expend.Add(btsInputShape[i]);
        }

        List<int> shape_shrink = new();
        shape_shrink.Add(shape_expend[btsBlockShape.Length]);
        for (var i = 0; i < btsBlockShape.Length; i++)
        {
            shape_shrink.Add(btsBlockShape[i] * btsInputShape[i + 1]);
        }

        for (var i = btsBlockShape.Length + 1; i < btsInputShape.Length; i++)
        {
            shape_shrink.Add(btsInputShape[i]);
        }

        List<int> crop_begs = new(), crop_ends = new();
        crop_begs.Add(0);
        crop_ends.Add(shape_shrink[0]);
        for (var i = 0; i < crop.GetLength(0); i++)
        {
            crop_begs.Add(crop[i, 0]);
            crop_ends.Add(shape_shrink[i + 1] - crop[i, 1]);
        }

        for (var i = btsBlockShape.Length + 1; i < btsInputShape.Length; i++)
        {
            crop_begs.Add(0);
            crop_ends.Add(shape_shrink[i]);
        }

        var cropBegin = crop_begs.ToArray();
        var cropEnd = crop_ends.ToArray();
        var strides = Enumerable.Repeat(1, crop_begs.Count).ToArray();
        var begin = NormalizeStridedSliceBegin(btsInputShape, cropBegin, strides, 0);
        var end = NormalizeStridedSliceEndEnd(btsInputShape, begin, cropEnd, strides, 0, 0);
        return (begin, end);
    }

    private int[] NormalizeStridedSliceEndEnd(int[] in_shape, int[] begin, int[] end, int[] strides, int end_mask, int shrink_axis_mask)
    {
        var new_shape = Enumerable.Range(0, strides.Length).ToArray();
        for (var i = 0; i < new_shape.Length; i++)
        {
            var stride = strides[i];
            var end_val = (end_mask & (1 << i)) != 0
                ? stride > 0 ? in_shape[i] : -1
                : (shrink_axis_mask & (1 << i)) == 0 ? (end[i] >= 0 ? end[i] : in_shape[i] + end[i] + 1)
                    : begin[i] + 1;
            new_shape[i] = end_val;
        }

        return new_shape;
    }

    private int[] NormalizeStridedSliceBegin(int[] in_shape, int[] begin, int[] strides, int begin_mask)
    {
        var new_shape = Enumerable.Range(0, strides.Length).ToArray();
        for (var i = 0; i < new_shape.Length; i++)
        {
            var stride = strides[i];
            new_shape[i] = (begin_mask & (1 << i)) != 0
                ? stride > 0 ? 0 : in_shape[i]
                : (begin[i] >= 0 ? begin[i] : in_shape[i] + begin[i]);
        }

        return new_shape;
    }
}
