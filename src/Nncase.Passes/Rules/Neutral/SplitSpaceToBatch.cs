// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
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

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class SplitSpaceToBatch : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSpaceToBatch(
        IsWildcard("input") with { TypePattern = HasRank() },
        IsWildcard("blockShape") with { TypePattern = HasFixedShape() },
        IsWildcard("paddings"));

    public Expr? GetReplace(Expr input, Expr blockShape, Expr paddings)
    {
        var spatialSize = blockShape.CheckedShape.Size;
        var remainShapeSize = input.CheckedShape.Rank - spatialSize - 1;
        var newPaddings = Enumerable.Repeat((Expr)0, (1 + spatialSize + remainShapeSize) * 2).ToArray();
        for (int i = 0; i < spatialSize; i++)
        {
            newPaddings[1 + i] = paddings[i, 0];
            newPaddings[1 + (newPaddings.Length / 2) + i] = paddings[i, 1];
        }

        var tmpPaddings = Stack(new IR.Tuple(newPaddings), 0);
        var newPaddingsTensor = Transpose(Reshape(tmpPaddings, new long[] { 2, 1 + spatialSize + remainShapeSize }), new long[] { 1, 0 });
        var p = Pad(NCHWToNHWC(input), newPaddingsTensor, PadMode.Constant, 0f);

        var padShape = Cast(ShapeOf(p), DataTypes.Int32);
        var batchShape1 = StackScalar(padShape[0]);
        var spatialShape1 = RangeExec(
                spatialSize,
                i => Stack(new IR.Tuple(padShape[i + 1] / blockShape[i], blockShape[i]), 0))
            .Aggregate((x, y) => Concat(new IR.Tuple(x, y), 0));
        var remainShape1 = Stack(new IR.Tuple(RangeExec(remainShapeSize, i => padShape[1 + spatialSize + i])), 0);
        var reshappedShape1 = Concat(
            new IR.Tuple(
            batchShape1,
            spatialShape1,
            remainShape1),
            0);

        var perm = RangeExec(spatialSize, i => (i * 2) + 2)
            .Concat(new[] { 0 })
            .Concat(RangeExec(spatialSize, i => (i * 2) + 1))
            .Concat(RangeExec(remainShapeSize, i => i + ((int)spatialSize * 2) + 1))
            .Select(x => (long)x)
            .ToArray();

        var reshappedShape2 = Concat(
            input: new IR.Tuple(
                StackScalar(padShape[0] * Prod(blockShape)),
                Stack(new IR.Tuple(RangeExec(spatialSize, i => padShape[i + 1] / blockShape[i])), 0),
                Stack(new IR.Tuple(RangeExec(remainShapeSize, i => padShape[1 + spatialSize + i])), 0)),
            0);

        var reshape1 = Reshape(p, reshappedShape1);
        var rt = Transpose(reshape1, perm);
        var reshape2 = Reshape(rt, reshappedShape2);
        return NHWCToNCHW(reshape2);
    }

    private T[] RangeExec<T>(long end, Func<int, T> f)
    {
        return EndRange(0, (int)end).Select(f).ToArray();
    }

    private IEnumerable<int> EndRange(int begin, int end)
    {
        return Enumerable.Range(begin, end - begin);
    }
}

[RuleGenerator]
public partial class SplitBatchToSpace : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBatchToSpace(
        IsWildcard("input") with { TypePattern = HasRank() },
        IsWildcard("blockShape") with { TypePattern = HasFixedShape() },
        IsWildcard("crop"));

    public Expr? GetReplace(Expr input, Expr blockShape, Expr crop)
    {
        // to nhwc
        var input0 = NCHWToNHWC(input);
        var blockLen = blockShape.CheckedShape.Size;
        var xLen = input0.CheckedShape.Rank;
        var xShape = Cast(ShapeOf(input0), DataTypes.Int32);
        var spatial = ShapeExprUtility.Slice(xShape, 1, blockLen + 1);
        var depth = ShapeExprUtility.Slice(xShape, blockLen + 1, xLen);
        var targetSpatial = spatial * blockShape;

        var ccat1 = Concat(new IR.Tuple(spatial, blockShape), 0);
        var re1 = Reshape(ccat1, new[] { ccat1.CheckedShape[0].FixedValue / blockLen, blockLen });
        var interLeave = Reshape(Transpose(re1, new long[] { 1, 0 }), new long[] { -1 });
        var shape1 = Concat(new IR.Tuple(new int[] { -1 }, interLeave, depth), 0);

        var g1 = BoostRange(2, (2 * blockLen) + 1, 2);
        var g2 = BoostRange(1, (2 * blockLen) + 1, 2);
        var g3 = BoostRange(0, xLen + blockLen).ToArray()[1 + (2 * blockLen)];
        var indices = g1.Append(0).Concat(g2).Append(g3);

        var perm = GetPerm(xLen, blockLen);

        var newShape = indices.Select(i => shape1[i]).ToArray();
        var x2 = Reshape(input0, Stack(new IR.Tuple(newShape), 0));
        var tr2 = Transpose(x2, perm);
        var shape2 = Concat(new IR.Tuple(new[] { -1 }, targetSpatial, depth), 0);
        var x3 = Reshape(tr2, shape2);

        var cropTransposed = Transpose(crop, new long[] { 1, 0 });
        var cropArray = Reshape(cropTransposed, new long[] { -1 });
        var w = cropTransposed.CheckedShape[1].FixedValue;
        var cropStart = ShapeExprUtility.Slice(cropArray, 0, w);
        var cropEnd = ShapeExprUtility.Slice(cropArray, w, w + w);
        var endRange = targetSpatial - cropEnd;
        var axesConst = BoostRange(1, blockLen + 1).ToArray();
        var strideConst = Enumerable.Repeat(1, axesConst.Length).ToArray();
        var result = Slice(x3, cropStart, endRange, axesConst, strideConst);

        // to nchw
        var transposeResult = NHWCToNCHW(result);
        return transposeResult;
    }

    private static IEnumerable<int> BoostRange(int start, int end, int step = 1)
    {
        int x = start;
        do
        {
            yield return x;
            x += step;
            if ((step < 0 && x <= end) || (step > 0 && end <= x))
            {
                break;
            }
        }
        while (true);
    }

    private long[] GetPerm(int xLen, int blockLen)
    {
        var perm = Enumerable.Range(0, xLen + blockLen).ToArray();
        perm[0] = blockLen;
        perm[1] = blockLen + 1;
        perm[2] = 0;
        foreach (var i in BoostRange(3, (blockLen * 2) + 1))
        {
            perm[i] = perm[i - 2] + 1;
        }

        return perm.Select(x => (long)x).ToArray();
    }

    private T[] ZipExec<T>(T[] a, T[] b, Func<T, T, T> f)
    {
        return a.Zip(b).Select(x => f(x.First, x.Second)).ToArray();
    }
}
