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
using OrtKISharp;
using Range = System.Range;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="SpaceToBatch"/>.
/// </summary>
public class SpaceToBatchEvaluator : IEvaluator<SpaceToBatch>, ITypeInferencer<SpaceToBatch>, ICostEvaluator<SpaceToBatch>, IMetricEvaluator<SpaceToBatch>
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
        return reshape2.ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, SpaceToBatch target)
    {
        var input = context.CheckArgumentType<TensorType>(target, SpaceToBatch.Input);
        var blockShape = context.CheckArgumentType<TensorType>(target, SpaceToBatch.BlockShape);
        var paddings = context.CheckArgumentType<TensorType>(target, SpaceToBatch.Paddings);
        return Visit(context, target, input, blockShape, paddings);
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
            var padded_shape = input.Shape.ToList();
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

            foreach (var i in Enumerable.Range(m + 1, padded_shape.Count - (m + 1)))
            {
                outshape.Add(padded_shape[i]);
            }

            foreach (var block in ts_block_shape)
            {
                outshape[0] = outshape[0].IsUnknown ? Dimension.Unknown : outshape[0].FixedValue * block;
            }

            return input with { Shape = new Shape(outshape) };
        }

        return new InvalidType("Can't Infer Shape With Dynamic Input!");
    }
}
