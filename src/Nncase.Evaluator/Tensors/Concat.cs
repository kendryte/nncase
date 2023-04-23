// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using Concat = Nncase.IR.Tensors.Concat;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Concat"/>.
/// </summary>
public class ConcatEvaluator : IEvaluator<Concat>, ITypeInferencer<Concat>, ICostEvaluator<Concat>,
    IShapeEvaluator<Concat>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Concat cat)
    {
        var inputs = context.GetArgumentValueAsTensors(cat, Concat.Input);
        var axis = context.GetArgumentValueAsScalar<int>(cat, Concat.Axis);
        return OrtKI.Concat(inputs.Select(t => t.ToOrtTensor()).ToArray(), axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Concat target)
    {
        var inputs = context.CheckArgumentType<TupleType>(target, Concat.Input);
        var axis = context.CheckArgumentType<TensorType>(target, Concat.Axis);
        return Visit(context, target, inputs, axis);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Concat target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret),
        };
    }

    private IRType? CheckType(TupleType inputs)
    {
        bool? allScalar = null;
        DataType? allDType = null;
        foreach (var (i, input) in Enumerable.Range(0, inputs.Count).Select(i => (i, inputs[i])))
        {
            var type = input as TensorType;
            if (type is null)
            {
                if (input is InvalidType)
                {
                    return input;
                }
                else
                {
                    return new InvalidType($"The ConCat Item[{i}] Must Be TensorType But Get {input.GetType().Name}");
                }
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

    private IRType Visit(ITypeInferenceContext context, Concat target, TupleType inputs, TensorType axis)
    {
        var result = CheckType(inputs);
        if (result != null)
        {
            return result;
        }

        var sameRank = inputs.All(input => ((TensorType)input).Shape.Rank == ((TensorType)inputs[0]).Shape.Rank);
        if (!sameRank)
        {
            return new InvalidType("Inputs of concat should be same rank");
        }
        var input0 = (TensorType)inputs[0];
        InvalidType? invalidType = null;
        var axisV = ((TensorConst)context.GetArgument(target, Concat.Axis)).Value.ToScalar<int>();
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
                    if (((TensorType)inType).Shape.IsUnranked)
                    {
                        continue;
                    }

                    var d = ((TensorType)inType).Shape[i];
                    if (d.IsUnknown)
                    {
                        return Dimension.Unknown;
                    }

                    if (d.FixedValue != ((TensorType)inputs[0]).Shape[i])
                    {
                        allAxisDimIsSame = false;
                    }
                }

                if (allAxisDimIsSame)
                {
                    return ((TensorType)inputs[0]).Shape[i];
                }
                else
                {
                    invalidType = new InvalidType("Concat dims that except the shape of axis dim are different");
                    return Dimension.Unknown;
                }
            }
        });
        var shape = new Shape(shapeValue);
        return (invalidType as IRType) ?? new TensorType(input0.DType, shape);
    }

    // axis: if one of inputs shape[axis] is unknown
    // then dims axis is known
    // else get sum of dims
    private Dimension AxisDim(TupleType inputs, int axisValue)
    {
        var allAxisDimIsFixed = inputs.Fields.Aggregate(
            true,
            (prod, next) => prod && ((TensorType)next).Shape[axisValue].IsFixed);
        if (allAxisDimIsFixed)
        {
            return inputs.Fields.Aggregate(
                0,
                (prod, next) => prod + ((TensorType)next).Shape[axisValue].FixedValue);
        }
        else
        {
            return Dimension.Unknown;
        }
    }

    public Expr Visit(IShapeEvaluateContext context, Concat target)
    {
        var inShape = context.GetArgumentShape(target, Concat.Input);
        var axis = context.GetArgument(target, Concat.Axis);
        var axisV = ShapeExprUtility.Positive(axis, inShape[0]);
        var inShapes = ((IR.Tuple)inShape).Fields;
        var dim = inShapes.ToArray().Aggregate((Expr)0, (sum, shape) => sum + shape[axisV]);
        var outShape = ShapeExprUtility.Replace(inShapes[0], axisV, dim);
        return outShape;
    }
}
