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
public class ReduceArgEvaluator : IEvaluator<ReduceArg>, ITypeInferencer<ReduceArg>, ICostEvaluator<ReduceArg>, IMetricEvaluator<ReduceArg>
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
        var input = context.CheckArgumentType<IRType>(target, ReduceArg.Input);
        return input switch
        {
            TensorType tensorType => Visit(context, target, tensorType),
            DistributedType distributedType => Visit(context, target, distributedType),
            _ => new InvalidType(string.Empty),
        };
    }

    public Cost Visit(ICostEvaluateContext context, ReduceArg target)
    {
        var input = context.GetArgumentType<IRType>(target, ReduceArg.Input);
        var ret = context.GetReturnType<IRType>();
        var inShape = input switch { TensorType t => t.Shape, DistributedType d => d.TensorType.Shape, _ => throw new NotImplementedException() };
        var rShape = ret switch { TensorType t => t.Shape, DistributedType d => d.TensorType.Shape, _ => throw new NotImplementedException() };
        uint input_elem = inShape.Aggregate(1U, (acc, d) => acc * (d.IsFixed ? (uint)d.FixedValue : 1U));
        uint ret_elem = rShape.Aggregate(1U, (acc, d) => acc * (d.IsFixed ? (uint)d.FixedValue : 1U));
        uint macPerElement = input_elem / ret_elem;
        return new() { [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input), [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret), [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, macPerElement), };
    }

    public Metric Visit(IMetricEvaluateContext context, ReduceArg target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, ReduceArg.Input);
        var returnType = context.GetReturnType<TensorType>();
        var rF = MetricUtility.GetFLOPs(returnType);
        var iF = MetricUtility.GetFLOPs(inputType);
        var inner = iF / rF;
        _ = iF / inner;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = iF,
        };
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

    private IRType Visit(ITypeInferenceContext context, ReduceArg target, DistributedType distributedType)
    {
        var rType = Visit(context, target, distributedType.TensorType);
        if (rType is not TensorType tensorType)
        {
            return rType;
        }

        if (!distributedType.NdSBP.All(sbp => sbp is SBPBroadCast))
        {
            return new InvalidType(string.Empty);
        }

        return distributedType with { TensorType = tensorType };
    }
}
