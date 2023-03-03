// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Reduce"/>.
/// </summary>
public class ReduceArgEvaluator : IEvaluator<ReduceArg>, ITypeInferencer<ReduceArg>, ICostEvaluator<ReduceArg>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ReduceArg reduceArg)
    {
        var input = context.GetOrtArgumentValue(reduceArg, ReduceArg.Input);
        var axis = context.GetArgumentValueAsScalar<long>(reduceArg, ReduceArg.Axis);
        var keepDims = context.GetArgumentValueAsScalar<long>(reduceArg, ReduceArg.KeepDims);
        var selectLastIndex = context.GetArgumentValueAsScalar<long>(reduceArg, ReduceArg.SelectLastIndex);
        var result = reduceArg.ReduceArgOp switch
        {
            ReduceArgOp.ArgMax => OrtKI.ArgMax(input, axis, keepDims, selectLastIndex),
            ReduceArgOp.ArgMin => OrtKI.ArgMin(input, axis, keepDims, selectLastIndex),
            _ => throw new ArgumentOutOfRangeException(nameof(reduceArg)),
        };

        if (reduceArg.DestType == DataTypes.Int32)
        {
            return result.Cast(OrtDataType.Int32).ToValue();
        }

        return result.ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ReduceArg target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ReduceArg.Input);
        return Visit(context, target, input);
    }

    public Cost Visit(ICostEvaluateContext context, ReduceArg target)
    {
        var input = context.GetArgumentType<TensorType>(target, ReduceArg.Input);
        var ret = context.GetReturnType<TensorType>();
        var input_elem = input.Shape.Aggregate(1, (acc, d) => acc * (d.IsFixed ? d.FixedValue : 1));
        var ret_elem = ret.Shape.Aggregate(1, (acc, d) => acc * (d.IsFixed ? d.FixedValue : 1));
        var macPerElement = input_elem / ret_elem;
        return new() { [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input), [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret), [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, macPerElement), };
    }

    private IRType Visit(ITypeInferenceContext context, ReduceArg target, TensorType input)
    {
        if (context.GetArgument(target, ReduceArg.Axis) is TensorConst axisValue &&
            context.GetArgument(target, ReduceArg.KeepDims) is TensorConst keepDimsValue)
        {
            var shape = input.Shape.ToList();
            var axisIndex = axisValue.Value.ToScalar<int>();
            axisIndex = axisIndex >= 0 ? axisIndex : input.Shape.Rank + axisIndex;
            if (keepDimsValue.Value.ToScalar<bool>())
            {
                shape[axisIndex] = 1;
            }
            else
            {
                shape.RemoveAt(axisIndex);
            }

            return input with { Shape = new Shape(shape), DType = target.DestType };
        }
        else
        {
            return new InvalidType("ReduceArg axis and keepDims are not const");
        }
    }
}
