// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using OrtKISharp;
using static OrtKISharp.TensorHelper;
using static Nncase.IR.F.Tensors;
namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="BatchToSpace"/>.
/// </summary>
public class BatchToSpaceEvaluator : IEvaluator<BatchToSpace>, ITypeInferencer<BatchToSpace>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, BatchToSpace s)
    {
        var input = context.GetOrtArgumentValue(s, BatchToSpace.Input);
        // to nhwc
        var input0 = OrtKI.Transpose(input, new long[] { 0, 2, 3, 1 });
        var blockShape = context.GetArgumentValueAsArray<int>(s, BatchToSpace.BlockShape);
        var crop = context.GetOrtArgumentValue(s, BatchToSpace.Crops);

        var blockLen = blockShape.Length;
        var xLen = input0.Rank;
        var xShape = input0.Shape;
        var spatial = xShape[1..(blockLen + 1)];
        var depth = xShape[(blockLen + 1)..xLen];
        var targetSpatial = ZipExec(spatial, blockShape, (x, y) => x * y);

        var ccat1 = spatial.Concat(blockShape).ToArray();
        var re1 = Tensor.FromSpan(ccat1, new[] { ccat1.Length / blockLen, blockLen });
        var interLeave = OrtKI.Transpose(re1.ToOrtTensor(), new long[] { 1, 0 }).ToArray<int>();
        var shape1 = new[] { -1 }.Concat(interLeave).Concat(depth).ToArray();

        var g1 = BoostRange(2, 2 * blockLen + 1, 2);
        var g2 = BoostRange(1, 2 * blockLen + 1, 2);
        var g3 = BoostRange(0, xLen + blockLen).ToArray()[1 + 2 * blockLen];
        var indices = g1.Append(0).Concat(g2).Append(g3);

        var perm = GetPerm(xLen, blockLen);

        var newShape = indices.Select(i => (long)shape1[i]).ToArray();
        var x2 = OrtKI.Reshape(input0, newShape, 0);
        var tr2 = OrtKI.Transpose(x2, perm);
        var shape2 = new[] { -1 }.Concat(targetSpatial).Concat(depth).Select(x => (long)x).ToArray();
        var x3 = OrtKI.Reshape(tr2, MakeOrtTensor(shape2), 0);

        var cropTransposed = OrtKI.Transpose(crop, new long[] { 1, 0 });
        var cropArray = cropTransposed.ToArray<int>();
        var w = cropTransposed.Shape[1];
        var cropStart = cropArray[..w];
        var cropEnd = cropArray[w..(w + w)];
        var endRange = ZipExec(targetSpatial, cropEnd, (x, y) => x - y);
        var axesConst = BoostRange(1, blockLen + 1).ToArray();
        var strideConst = Enumerable.Repeat(1, axesConst.Length).ToArray();
        var result = OrtKI.Slice(x3, cropStart, endRange, axesConst, strideConst);
        // to nchw
        var transposeResult = OrtKI.Transpose(result, new long[] { 0, 3, 1, 2 });
        return transposeResult.ToValue();
    }

    private T[] ZipExec<T>(T[] a, T[] b, Func<T, T, T> f)
    {
        return a.Zip(b).Select(x => f(x.Item1, x.Item2)).ToArray();
    }

    private long[] GetPerm(int xLen, int blockLen)
    {
        var perm = Enumerable.Range(0, xLen + blockLen).ToArray();
        perm[0] = blockLen;
        perm[1] = blockLen + 1;
        perm[2] = 0;
        foreach (var i in BoostRange(3, blockLen * 2 + 1))
        {
            perm[i] = perm[i - 2] + 1;
        }

        return perm.Select(x => (long)x).ToArray();
    }

    private static IEnumerable<int> BoostRange(int start, int end, int step = 1)
    {
        int x = start;
        do
        {
            yield return x;
            x += step;
            if (step < 0 && x <= end || 0 < step && end <= x)
                break;
        }
        while (true);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BatchToSpace target)
    {
        var input = context.CheckArgumentType<TensorType>(target, BatchToSpace.Input);
        var blockShape = context.CheckArgumentType<TensorType>(target, BatchToSpace.BlockShape);
        var crops = context.CheckArgumentType<TensorType>(target, BatchToSpace.Crops);
        return Visit(context, target, input, blockShape, crops);
    }

    private IRType Visit(ITypeInferenceContext context, BatchToSpace target, TensorType input, TensorType blockShape, TensorType crops)
    {
        // todo:
        var inShape = TypeInference.ApplyPerm(input.Shape, new[] {0, 2, 3, 1});
        var batch = inShape[0];
        if (context.GetArgument(target, BatchToSpace.BlockShape) is TensorConst blockShapeValue &&
            context.GetArgument(target, BatchToSpace.Crops) is TensorConst cropsValue)
        {
            if (crops.Shape.Rank != 2)
            {
                return new InvalidType("BatchToSpace crops rank must be 2");
            }

            var blockShapeArr = blockShapeValue.Value.ToArray<int>();
            var blockSize = blockShapeArr.Aggregate(1, (a, b) => a * b);
            var d0 = batch / blockSize;
            Debug.Assert(blockShape.Shape[0] == crops.Shape[0]);
            var M = blockShape.Shape[0].FixedValue;
            var cropsV = cropsValue.Value.Cast<int>();
            var cropSection = Enumerable.Range(0, M).Select(
                i => (inShape[i + 1] * blockShapeArr[0]) - cropsV[i, 0] - cropsV[i, 1]);
            
            var remainSize = inShape.Rank - 1 - M;
            var remainShape = remainSize > 0 ? inShape.Skip(1 + M) : new Dimension[] { };
            var outShapeList = new[] {d0}.Concat(cropSection).Concat(remainShape).ToArray();
            var outShape = TypeInference.ApplyPerm(outShapeList, new[] {0, 3, 1, 2});
            return input with {Shape = outShape};
        }
        else
        {
            return new InvalidType("BatchToSpace can't infer shape with dynamic crops");
        }
    }
}
