// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using Range = System.Range;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="SpaceToBatch"/>.
/// </summary>
public class SpaceToBatchEvaluator : IEvaluator<SpaceToBatch>, ITypeInferencer<SpaceToBatch>, ICostEvaluator<SpaceToBatch>, IMetricEvaluator<SpaceToBatch>, IShapeEvaluator<SpaceToBatch>
{
    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, SpaceToBatch target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, SpaceToBatch target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(ret) * 2,
        };
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, SpaceToBatch s)
    {
        var input = context.GetOrtArgumentValue(s, SpaceToBatch.Input);
        input = NCHWToNHWC(input.ToTensor()).Evaluate().AsTensor().ToOrtTensor();
        var blockShape = context.GetArgumentValueAsTensor<long>(s, SpaceToBatch.BlockShape);
        var paddings = context.GetArgumentValueAsArray<long>(s, SpaceToBatch.Paddings);
        var spatialSize = blockShape.Length;
        var remainShapeSize = input.Rank - spatialSize - 1;
        var newPaddings = new long[(1 + spatialSize + remainShapeSize) * 2];
        for (int i = 0; i < spatialSize; i++)
        {
            newPaddings[1 + i] = paddings[2 * i];
            newPaddings[1 + (newPaddings.Length / 2) + i] = paddings[(2 * i) + 1];
        }

        var newPaddingsTensor = (OrtKISharp.Tensor)newPaddings;
        var p = OrtKI.Pad(input, newPaddingsTensor, OrtKISharp.Tensor.FromScalar(0f), "constant");

        var batchShape1 = new long[] { p.Shape[0] };
        var spatialShape1 = RangeExec(
            spatialSize,
            i => new[] { p.Shape[i + 1] / blockShape[i], blockShape[i] })
            .Aggregate(Array.Empty<long>(), (x, y) => x.Concat(y).ToArray());
        var remainShape1 = RangeExec(remainShapeSize, i => (long)p.Shape[1 + spatialSize + i]);
        var reshappedShape1 = batchShape1.Concat(spatialShape1.Concat(remainShape1)).ToArray();

        var perm = RangeExec(spatialSize, i => (i * 2) + 2)
            .Concat(new[] { 0 })
            .Concat(RangeExec(spatialSize, i => (i * 2) + 1))
            .Concat(RangeExec(remainShapeSize, i => i + ((int)spatialSize * 2) + 1))
            .Select(x => (long)x)
            .ToArray();

        var reshappedShape2 = new[] { p.Shape[0] * blockShape.Aggregate(1L, (x, y) => x * y) }
            .Concat(RangeExec(spatialSize, i => p.Shape[i + 1] / blockShape[i]))
            .Concat(RangeExec(remainShapeSize, i => (long)p.Shape[1 + spatialSize + i]))
            .ToArray();

        var reshape1 = OrtKI.Reshape(p, (OrtKISharp.Tensor)reshappedShape1, 0);
        var rt = OrtKI.Transpose(reshape1, perm);
        var reshape2 = OrtKI.Reshape(rt, (OrtKISharp.Tensor)reshappedShape2, 0);

        return NHWCToNCHW(reshape2.ToTensor()).Evaluate();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, SpaceToBatch target)
    {
        var input = context.CheckArgumentType<TensorType>(target, SpaceToBatch.Input);
        var blockShape = context.CheckArgumentType<TensorType>(target, SpaceToBatch.BlockShape);
        var paddings = context.CheckArgumentType<TensorType>(target, SpaceToBatch.Paddings);
        return Visit(context, target, input, blockShape, paddings);
    }

    public Expr Visit(IShapeEvaluateContext context, SpaceToBatch target)
    {
        var inShape = context.GetArgumentShape(target, SpaceToBatch.Input);
        var inputExpr = context.GetArgument(target, SpaceToBatch.Input);
        if (inputExpr.CheckedShape.Rank == 4)
        {
            inShape = Stack(new IR.Tuple(new[] { inShape[0], inShape[2], inShape[3], inShape[1] }), 0);
        }
        else if (inputExpr.CheckedShape.Rank == 3)
        {
            inShape = Stack(new IR.Tuple(new[] { inShape[0], inShape[2], inShape[1] }), 0);
        }
        else
        {
            throw new InvalidOperationException();
        }

        var blockShape = context.GetArgument(target, SpaceToBatch.BlockShape);
        var padding = Cast(context.GetArgument(target, SpaceToBatch.Paddings), DataTypes.Int64);
        var input = context.GetArgument(target, SpaceToBatch.Input);
        if (blockShape is TensorConst blockConst)
        {
            var blockShapeValue = blockConst.Value.ToArray<long>();
            var m = blockShapeValue.Length;
            var inRank = input.CheckedShape.Rank;

            var paddedShape = new[] { inShape[0] }
                .Concat(Enumerable.Range(0, inRank)
                .Select(i =>
                {
                    return inShape[i + 1] + padding[2 * i, 0] + padding[2 * i, 1];
                }))
                .ToArray();
            var outFirst = new[] { paddedShape[0] * IR.F.Tensors.Prod(blockShapeValue) };

            // var inRank = Cast(ShapeOf(inShape)[0], DataTypes.Int32);
            var outMid = Enumerable.Range(0, m).Select(i =>
            {
                return paddedShape[i + 1] / blockShapeValue[i];
            }).ToArray();

            var remainSize = inRank - 1 - m;
            var remainShape = new If(remainSize > 0, ShapeExprUtility.Slice(inShape, 1 + m, int.MaxValue), Array.Empty<long>());
            var outLast = remainShape;
            var outShape = Concat(new IR.Tuple(Stack(new IR.Tuple(outFirst.Concat(outMid).ToArray()), 0), outLast), 0);

            if (inputExpr.CheckedShape.Rank == 4)
            {
                outShape = Stack(new IR.Tuple(new[] { outShape[0], outShape[3], outShape[1], outShape[2] }), 0);
            }
            else if (inputExpr.CheckedShape.Rank == 3)
            {
                outShape = Stack(new IR.Tuple(new[] { outShape[0], outShape[2], outShape[1] }), 0);
            }
            else
            {
                throw new InvalidOperationException();
            }

            return outShape;
        }

        throw new NotImplementedException();
    }

    private T[] RangeExec<T>(long end, Func<int, T> f)
    {
        return EndRange(0, (int)end).Select(f).ToArray();
    }

    private IEnumerable<int> EndRange(int begin, int end)
    {
        return Enumerable.Range(begin, end - begin);
    }

    private IRType Visit(ITypeInferenceContext context, SpaceToBatch target, TensorType input, TensorType blockShape, TensorType paddings)
    {
        if (context.GetArgument(target, SpaceToBatch.BlockShape) is TensorConst block_shape_con &&
             context.GetArgument(target, SpaceToBatch.Paddings) is TensorConst paddings_con)
        {
            var ts_block_shape = block_shape_con.Value.Cast<int>();
            var ts_paddings = paddings_con.Value.ToArray<int>();
            int m = (int)ts_block_shape.Length;

            // var padded_shape = input.Shape.ToList();
            var inShape = input.Shape.ToList();
            Dimension[] padded_shape;

            // nchw to nhwc
            if (inShape.Count == 4)
            {
                padded_shape = new[] { inShape[0], inShape[2], inShape[3], inShape[1] };
            }
            else if (inShape.Count == 3)
            {
                padded_shape = new[] { inShape[0], inShape[2], inShape[1] };
            }
            else
            {
                throw new InvalidOperationException();
            }

            for (int i = 0; i < m; i++)
            {
                if (!padded_shape[1 + i].IsUnknown)
                {
                    padded_shape[1 + i] += new Dimension(ts_paddings[2 * i] + ts_paddings[(2 * i) + 1]);
                }
            }

            var outshape = new List<Dimension> { padded_shape[0] };
            foreach (var i in Enumerable.Range(1, m))
            {
                outshape.Add(padded_shape[i].IsUnknown ? Dimension.Unknown :
                                    padded_shape[i].FixedValue % ts_block_shape[i - 1] == 0 ?
                                      padded_shape[i].FixedValue / ts_block_shape[i - 1] :
                                      throw new TypeInferenceInterruptException(
                                        new InvalidType($"The Padded Shape Must Divides BlockShape!")));
            }

            foreach (var i in Enumerable.Range(m + 1, padded_shape.Length - (m + 1)))
            {
                outshape.Add(padded_shape[i]);
            }

            foreach (var block in ts_block_shape)
            {
                outshape[0] = outshape[0].IsUnknown ? Dimension.Unknown : outshape[0].FixedValue * block;
            }

            // return input with { Shape = new Shape(outshape) };
            Dimension[] outputShape;

            // nhwc to nchw
            if (inShape.Count == 4)
            {
                outputShape = new[] { outshape[0], outshape[3], outshape[1], outshape[2] };
            }
            else
            {
                outputShape = new[] { inShape[0], inShape[2], inShape[1] };
            }

            return input with { Shape = new Shape(outputShape) };
        }

        return new TensorType(input.DType, Enumerable.Repeat(Dimension.Unknown, input.Shape.Count).ToArray());
    }
}
