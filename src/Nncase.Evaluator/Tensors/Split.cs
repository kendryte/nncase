// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Split"/>.
/// </summary>
public class SplitEvaluator : IEvaluator<Split>, ITypeInferencer<Split>, ICostEvaluator<Split>, IMetricEvaluator<Split>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Split target)
    {
        var input = context.GetOrtArgumentValue(target, Split.Input);
        var split = context.GetInt64OrtTensorArgumentValue(target, Split.Sections);
        var axis = context.GetArgumentValueAsScalar<long>(target, Split.Axis);
        var result = OrtKI.Split(input, split, axis);
        return Value.FromTensors(result.Select(t => t.ToTensor()).ToArray());
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Split target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Split.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Split target)
    {
        // _ = context.GetReturnType<TupleType>();
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Split target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Split.Input);
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Split target, TensorType input)
    {
        var inShape = input.Shape as RankedShape;
        if (context.GetArgument(target, Split.Axis) is DimConst axis_con &&
            context.GetArgument(target, Split.Sections) is RankedShape { IsFixed: true } sections_con)
        {
            var axis_v = Util.PositiveIndex((int)axis_con.Value, input.Shape.Rank);
            var sections_v = sections_con.ToValueArray();

            if (inShape is null)
            {
                return new TupleType(Enumerable.Repeat((IRType)(input with { Shape = Shape.Unranked }), (int)sections_v.Length));
            }

            var inshape = inShape.ToArray();

            // split
            if (sections_v.Length == 1)
            {
                if (inshape[axis_v].FixedValue % sections_v[0] != 0)
                {
                    return new InvalidType("The Section Value Not Match Shape[Axis]!");
                }

                var outshape = new Dimension[inshape.Length];
                Array.Copy(inshape, outshape, inshape.Length);
                outshape[axis_v] = inshape[axis_v].FixedValue / sections_v[0];
                return new TupleType(Enumerable.Repeat((IRType)(input with { Shape = new RankedShape(outshape) }), (int)sections_v[0]));
            }
            else
            {
                if (inshape[axis_v].IsFixed && sections_v.Sum() != inshape[axis_v].FixedValue)
                {
                    return new InvalidType("The Sections Sum Must Equal To Shape[Axis]!");
                }

                var outshape = new Dimension[inshape.Length];
                Array.Copy(inshape, outshape, inshape.Length);
                return new TupleType(from section in sections_v
                                     let x = outshape[axis_v] = section
                                     select input with { Shape = new RankedShape(outshape) });
            }
        }

        if (inShape is null)
        {
            return new TupleType(new IRType[] { input with { Shape = Shape.Unranked } }, IsVariadic: true);
        }

        var splitedShape = inShape.ToArray();
        if (context.GetArgument(target, Split.Axis) is DimConst axisCon)
        {
            var axisV = Util.PositiveIndex((int)axisCon.Value, input.Shape.Rank);
            splitedShape[axisV] = Dimension.Unknown;
        }
        else
        {
            splitedShape = Enumerable.Repeat(Dimension.Unknown, splitedShape.Rank).ToArray();
        }

        return new TupleType(new IRType[] { input with { Shape = splitedShape } }, IsVariadic: true);
    }
}
