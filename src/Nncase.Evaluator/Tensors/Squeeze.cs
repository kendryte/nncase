// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Squeeze"/>.
/// </summary>
public class SqueezeEvaluator : IEvaluator<Squeeze>, ITypeInferencer<Squeeze>, ICostEvaluator<Squeeze>, IShapeEvaluator<Squeeze>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Squeeze squeeze)
    {
        var input = context.GetOrtArgumentValue(squeeze, Squeeze.Input);
        var dims = context.GetInt64OrtTensorArgumentValue(squeeze, Squeeze.Dim);
        return OrtKI.Squeeze(input, dims).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Squeeze target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Squeeze.Input);
        return Visit(context, target, input);
    }

    public Cost Visit(ICostEvaluateContext context, Squeeze target)
    {
        return CostUtility.GetReshapeCost();
    }

    public Expr Visit(IShapeEvaluateContext context, Squeeze target)
    {
        var inShape = context.GetArgumentShape(target, Squeeze.Input);
        var input = context.GetArgument(target, Squeeze.Input);
        var dims = context.GetArgument(target, Squeeze.Dim);
        if (dims is TensorConst dimConst)
        {
            var dimValue = dimConst.Value.ToArray<int>();
            var rank = input.CheckedShape.Count;
            var outDims = Enumerable.Range(0, rank).Where(i => !dimValue.Contains(i)).Select(i => inShape[i]).ToArray();
            if (outDims.Length == 0)
            {
                return 1;
            }

            return IR.F.Tensors.Stack(new IR.Tuple(outDims), 0);
        }

        throw new NotImplementedException();
    }

    private IRType Visit(ITypeInferenceContext context, Squeeze target, TensorType input)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (context.GetArgument(target, Squeeze.Dim) is TensorConst dim_con)
        {
            var dims = dim_con.Value.Cast<int>();
            var outshape = input.Shape.ToList();
            if (dims.Length == 0)
            {
                return input with { Shape = new Shape(outshape.Where(x => x != 1).ToArray()) };
            }

            foreach (var dimV in dims)
            {
                var dimValue = Util.PositiveIndex(dimV, input.Shape.Rank);
                outshape[dimValue] = int.MaxValue;
            }

            return input with { Shape = new Shape(outshape.Where(x => x != int.MaxValue)) };
        }

        return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Count - 1)) };
    }
}
