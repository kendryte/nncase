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
using static Nncase.IR.F.Tensors;
using Concat = Nncase.IR.Tensors.Concat;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Concat"/>.
/// </summary>
public class ConcatEvaluator : IEvaluator<Concat>, ITypeInferencer<Concat>, ICostEvaluator<Concat>,
    IShapeEvaluator<Concat>, IMetricEvaluator<Concat>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Concat cat)
    {
        var inputs = context.GetArgumentValueAsTensors(cat, Concat.Input);
        var axis = cat.Axis;
        return OrtKI.Concat(inputs.Select(t => t.ToOrtTensor()).ToArray(), axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Concat target)
    {
        var inputs = context.CheckArgumentType<TupleType>(target, Concat.Input);
        return Visit(inputs, target.Axis);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Concat target)
    {
        var ret = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Concat target)
    {
        var inShape = context.GetArgumentShape(target, Concat.Input);
        var axisV = ShapeExprUtility.Positive(target.Axis, inShape[0]);
        var inShapes = ((IR.Tuple)inShape).Fields.ToArray().Select(x => Cast(x, DataTypes.Int64)).ToArray();
        var dim = inShapes.ToArray().Aggregate((Expr)0L, (sum, shape) => sum + shape[axisV]);
        var outShape = ShapeExprUtility.Replace(inShapes[0], axisV, dim);
        return outShape;
    }

    public Metric Visit(IMetricEvaluateContext context, Concat target) => Metric.Zero;

    private IRType? CheckType(TupleType inputs)
    {
        bool? allScalar = null;
        DataType? allDType = null;
        foreach (var (i, input) in Enumerable.Range(0, inputs.Count).Select(i => (i, inputs[i])))
        {
            TensorType type;
            if (input is TensorType a)
            {
                type = a;
            }
            else if (input is DistributedType { TensorType: TensorType b })
            {
                type = b;
            }
            else
            {
                return new InvalidType($"The ConCat Item[{i}] Must Have TensorType But Get {input}");
            }

            if (type.Shape.IsUnranked)
            {
                return new TensorType(type.DType, Shape.Unranked);
            }

            allScalar = (allScalar ?? type.IsScalar) & type.IsScalar;
            allDType ??= type.DType;
            if (allDType != type.DType)
            {
                return new InvalidType(
                    $"The ConCat Item[{i}] Must Be {allDType} But Get {type.DType.GetDisplayName()}");
            }
        }

        if (allScalar == true && allDType is not null)
        {
            return new TensorType(allDType, new[] { inputs.Count });
        }

        return null;
    }

    private TensorType GetTensorType(IRType input) => input switch
    {
        TensorType t => t,
        DistributedType d => d.TensorType,
        _ => throw new InvalidCastException(),
    };

    private IRType Visit(TupleType inputs, int axis)
    {
        var result = CheckType(inputs);
        if (result != null)
        {
            return result;
        }

        var sameRank = inputs.All(input => GetTensorType(input).Shape.Rank == GetTensorType(inputs[0]).Shape.Rank);
        if (!sameRank)
        {
            return new InvalidType("Inputs of concat should be same rank");
        }

        var input0 = GetTensorType(inputs[0]);
        InvalidType? invalidType = null;
        var axisV = axis;
        var axisValue = Util.PositiveIndex(axisV, input0.Shape.Rank);
        var shapeValue = Enumerable.Range(0, input0.Shape.Rank).Select(i =>
        {
            if (i == axisValue)
            {
                return AxisDim(inputs, axisValue);
            }

            // if all input shape[dim] is not same, return invalid
            else
            {
                var allAxisDimIsSame = true;
                foreach (var inType in inputs.Fields)
                {
                    if (GetTensorType(inType).Shape.IsUnranked)
                    {
                        continue;
                    }

                    var d = GetTensorType(inType).Shape[i];
                    if (d.IsUnknown)
                    {
                        return Dimension.Unknown;
                    }

                    if (d.FixedValue != GetTensorType(inputs[0]).Shape[i])
                    {
                        allAxisDimIsSame = false;
                    }
                }

                if (allAxisDimIsSame)
                {
                    return GetTensorType(inputs[0]).Shape[i];
                }
                else
                {
                    invalidType = new InvalidType("Concat dims that except the shape of axis dim are different");
                    return Dimension.Unknown;
                }
            }
        });
        var shape = new Shape(shapeValue);
        if (invalidType is InvalidType invalid)
        {
            return invalid;
        }

        var tensorType = new TensorType(input0.DType, shape);

        if (inputs[0] is not DistributedType distributedType)
        {
            return tensorType;
        }

        if (inputs.OfType<DistributedType>().Select(d => d.Placement).ToHashSet().Count != 1)
        {
            return new InvalidType("the inputs have different placement");
        }

        var ndsbp = new SBP[distributedType.Placement.Rank];

        for (int i = 0; i < distributedType.Placement.Rank; i++)
        {
            var sbps = inputs.OfType<DistributedType>().Select(d => d.NdSBP[i]).ToArray();
            if (sbps.Any(sbp => sbp is SBPSplit { Axis: int x } && x == axis))
            {
                return new InvalidType("not support distribute on concat axis");
            }

            if (sbps.Any(sbp => sbp is SBPPartialSum))
            {
                return new InvalidType("not support distribute with partialsum");
            }

            if (sbps.OfType<SBPSplit>().ToHashSet() is HashSet<SBPSplit> setSplit &&
            sbps.OfType<SBPSplit>().ToHashSet() is HashSet<SBPSplit> setBroadcast)
            {
                switch (setSplit.Count)
                {
                    case 0:
                        ndsbp[i] = SBP.B;
                        break;
                    case 1 when setBroadcast.Count == 0:
                        ndsbp[i] = setSplit.First();
                        break;
                    default:
                        return new InvalidType("not support distribute with different axis");
                }
            }
        }

        return new DistributedType(tensorType, ndsbp, distributedType.Placement);
    }

    // axis: if one of inputs shape[axis] is unknown
    // then dims axis is known
    // else get sum of dims
    private Dimension AxisDim(TupleType inputs, int axisValue)
    {
        var allAxisDimIsFixed = inputs.Fields.Aggregate(
            true,
            (prod, next) => prod && (next switch { TensorType t => t, DistributedType d => d.TensorType, _ => throw new NotSupportedException() }).Shape[axisValue].IsFixed);
        if (allAxisDimIsFixed)
        {
            return inputs.Fields.Aggregate(
                0,
                (prod, next) => prod + (next switch { TensorType t => t, DistributedType d => d.TensorType, _ => throw new NotSupportedException() }).Shape[axisValue].FixedValue);
        }
        else
        {
            return Dimension.Unknown;
        }
    }
}
