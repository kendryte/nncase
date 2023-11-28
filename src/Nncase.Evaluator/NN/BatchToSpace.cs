// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="BatchToSpace"/>.
/// </summary>
public class BatchToSpaceEvaluator : IEvaluator<BatchToSpace>, ITypeInferencer<BatchToSpace>, ICostEvaluator<BatchToSpace>, IMetricEvaluator<BatchToSpace>, IShapeEvaluator<BatchToSpace>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, BatchToSpace s)
    {
        var input = context.GetArgumentValue(s, BatchToSpace.Input);

        // to nhwc
        var input0 = NCHWToNHWC(input.AsTensor()).Evaluate().AsTensor().ToOrtTensor();
        var blockShape = context.GetArgumentValueAsArray<int>(s, BatchToSpace.BlockShape);
        var crop = context.GetOrtArgumentValue(s, BatchToSpace.Crops).Cast(OrtDataType.Int32);

        var blockLen = blockShape.Length;
        var xLen = input0.Rank;
        var xShape = input0.Shape.ToInts();
        var spatial = xShape[1..(blockLen + 1)];
        var depth = xShape[(blockLen + 1)..xLen];
        var targetSpatial = ZipExec(spatial, blockShape, (x, y) => x * y);

        var ccat1 = spatial.Concat(blockShape).ToArray();
        var re1 = Tensor.From(ccat1, new[] { ccat1.Length / blockLen, blockLen });
        var interLeave = OrtKI.Transpose(re1.ToOrtTensor(), new long[] { 1, 0 }).ToArray<int>();
        var shape1 = new int[] { -1 }.Concat(interLeave).Concat(depth).ToArray();

        var g1 = BoostRange(2, (2 * blockLen) + 1, 2);
        var g2 = BoostRange(1, (2 * blockLen) + 1, 2);
        var g3 = BoostRange(0, xLen + blockLen).ToArray()[1 + (2 * blockLen)];
        var indices = g1.Append(0).Concat(g2).Append(g3);

        var perm = GetPerm(xLen, blockLen);

        var newShape = indices.Select(i => (long)shape1[i]).ToArray();
        var x2 = OrtKI.Reshape(input0, newShape, 0);
        var tr2 = OrtKI.Transpose(x2, perm);
        var shape2 = new[] { -1 }.Concat(targetSpatial).Concat(depth).Select(x => (long)x).ToArray();
        var x3 = OrtKI.Reshape(tr2, shape2, 0);

        var cropTransposed = OrtKI.Transpose(crop, new long[] { 1, 0 });
        var cropArray = cropTransposed.ToArray<int>();
        var w = (int)cropTransposed.Shape[1];
        var cropStart = cropArray[..w];
        var cropEnd = cropArray[w..(w + w)];
        var endRange = ZipExec(targetSpatial, cropEnd, (x, y) => x - y);
        var axesConst = BoostRange(1, blockLen + 1).ToArray();
        var strideConst = Enumerable.Repeat(1, axesConst.Length).ToArray();
        var result = OrtKI.Slice(x3, cropStart, endRange, axesConst, strideConst);

        // to nchw
        var transposeResult = NHWCToNCHW(result.ToTensor()).Evaluate();
        return transposeResult;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BatchToSpace target)
    {
        var input = context.CheckArgumentType<TensorType>(target, BatchToSpace.Input);
        var blockShape = context.CheckArgumentType<TensorType>(target, BatchToSpace.BlockShape);
        var crops = context.CheckArgumentType<TensorType>(target, BatchToSpace.Crops);
        return Visit(context, target, input, blockShape, crops);
    }

    public Cost Visit(ICostEvaluateContext context, BatchToSpace target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, BatchToSpace.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, BatchToSpace target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, BatchToSpace.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, BatchToSpace target)
    {
        var inShape = context.GetArgumentShape(target, BatchToSpace.Input);
        var input = context.GetArgument(target, BatchToSpace.Input);
        if (input.CheckedShape.Rank == 4)
        {
            inShape = Stack(new IR.Tuple(inShape[0], inShape[2], inShape[3], inShape[1]), 0);
        }

        if (input.CheckedShape.Rank == 3)
        {
            inShape = Stack(new IR.Tuple(inShape[0], inShape[2], inShape[1]), 0);
        }

        var blockShape = Cast(context.GetArgument(target, BatchToSpace.BlockShape), DataTypes.Int64);
        if (!blockShape.CheckedShape.IsFixed)
        {
            throw new NotImplementedException();
        }

        var crops = Cast(context.GetArgument(target, BatchToSpace.Crops), DataTypes.Int64);
        var blockSize = Prod(blockShape);
        var batch = inShape[0];
        var d0 = batch / blockSize;
        var m = blockShape.CheckedShape[0].FixedValue;
        var cropSection = Enumerable.Range(0, m).Select(
            i => (inShape[i + 1] * blockShape[0]) - crops[i, 0] - crops[i, 1]).ToArray();

        var inRank = Cast(ShapeOf(inShape)[0], DataTypes.Int32);
        var remainSize = inRank - 1 - m;
        var remainShape = new If(remainSize > 0, ShapeExprUtility.Slice(inShape, 1 + m, int.MaxValue), Array.Empty<long>());

        var outShapeList = Concat(new IR.Tuple(Stack(new IR.Tuple(new[] { d0 }), 0), Stack(new IR.Tuple(cropSection), 0), remainShape), 0);

        if (input.CheckedShape.Rank == 4)
        {
            return Stack(new IR.Tuple(outShapeList[0], outShapeList[3], outShapeList[1], outShapeList[2]), 0);
        }

        if (input.CheckedShape.Rank == 3)
        {
            return Stack(new IR.Tuple(outShapeList[0], outShapeList[2], outShapeList[1]), 0);
        }

        throw new NotImplementedException();
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

    private T[] ZipExec<T>(T[] a, T[] b, Func<T, T, T> f)
    {
        return a.Zip(b).Select(x => f(x.First, x.Second)).ToArray();
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

    private IRType Visit(ITypeInferenceContext context, BatchToSpace target, TensorType input, TensorType blockShape, TensorType crops)
    {
        var inShape = input.Shape.Rank == 4
            ? TypeInference.ApplyPerm(input.Shape, new[] { 0, 2, 3, 1 })
            : TypeInference.ApplyPerm(input.Shape, new[] { 0, 2, 1 });
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
            Trace.Assert(blockShape.Shape[0] == crops.Shape[0]);
            var m = blockShape.Shape[0].FixedValue;
            var cropsV = cropsValue.Value.Cast<int>();
            var cropSection = Enumerable.Range(0, m).Select(
                i => (inShape[i + 1] * blockShapeArr[i]) - cropsV[i, 0] - cropsV[i, 1]);

            var remainSize = inShape.Rank - 1 - m;
            var remainShape = remainSize > 0 ? inShape.Skip(1 + m) : Array.Empty<Dimension>();
            var outShapeList = new[] { d0 }.Concat(cropSection).Concat(remainShape).ToArray();
            var outShape =
                outShapeList.Length == 4
                ? TypeInference.ApplyPerm(outShapeList, new[] { 0, 3, 1, 2 })
                : TypeInference.ApplyPerm(outShapeList, new[] { 0, 2, 1 });
            return input with { Shape = outShape };
        }
        else
        {
            return new TensorType(input.DType, Enumerable.Repeat(Dimension.Unknown, input.Shape.Count).ToArray());
        }
    }
}
