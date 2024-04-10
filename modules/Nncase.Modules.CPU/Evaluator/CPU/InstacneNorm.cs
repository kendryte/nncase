// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.CPU;

public sealed class InstanceNormEvaluator : IEvaluator<InstacneNorm>, ITypeInferencer<InstacneNorm>, ICostEvaluator<InstacneNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, InstacneNorm target)
    {
        var input = context.GetOrtArgumentValue(target, InstacneNorm.Input);
        var scale = context.GetOrtArgumentValue(target, InstacneNorm.Scale);
        var bias = context.GetOrtArgumentValue(target, InstacneNorm.Bias);
        if (target.PackedAxes.Count == 0)
        {
            return Value.FromTensor(OrtKI.InstanceNormalization(input, scale, bias, target.Epsilon).ToTensor());
        }
        else
        {
            var lanes = input.Shape.TakeLast(target.PackedAxes.Count).Select(i => (int)i).ToArray();
            var channelPadNums = 0;
            for (int i = target.PackedAxes.Count - 1; i >= 0; i--)
            {
                var axis = target.PackedAxes[i];
                if (axis == 1)
                {
                    channelPadNums = target.PadedNums[i];
                }

                input = input.Unpack(axis);
            }

            if (channelPadNums > 0)
            {
                var rk = target.PackedAxes.Count;
                var inshape = input.Shape.ToArray();
                input = OrtKI.Slice(
                    input,
                    target.PackedAxes.Select(_ => 0L).ToArray(),
                    Enumerable.Range(0, rk).Select(i => inshape[target.PackedAxes[i]] - target.PadedNums[i]).ToArray(),
                    target.PackedAxes.Select(axis => (long)axis).ToArray(),
                    target.PackedAxes.Select(_ => 1L).ToArray());
            }

            if (scale.Shape.Length == 2)
            {
                scale = scale.Unpack(0);
                if (channelPadNums > 0)
                {
                    scale = OrtKI.Slice(scale, new[] { 0L }, new[] { scale.Shape[0] - channelPadNums }, new[] { 0L }, new[] { 1L });
                }
            }

            if (bias.Shape.Length == 2)
            {
                bias = bias.Unpack(0);
                if (channelPadNums > 0)
                {
                    bias = OrtKI.Slice(bias, new[] { 0L }, new[] { bias.Shape[0] - channelPadNums }, new[] { 0L }, new[] { 1L });
                }
            }

            var norm = OrtKI.InstanceNormalization(input, scale, bias, target.Epsilon);
            var output = CPUEvaluatorUtility.RepackTensor(norm, lanes, target.PackedAxes, target.PadedNums);
            return Value.FromTensor(Tensor.FromBytes(new TensorType(new VectorType(norm.DataType.ToDataType(), lanes), output.Shape.SkipLast(target.PackedAxes.Count).Select(i => (int)i).ToArray()), output.BytesBuffer.ToArray()));
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, InstacneNorm target)
    {
        var input = context.CheckArgumentType<IRType>(target, InstacneNorm.Input);
        var scale = context.CheckArgumentType<IRType>(target, InstacneNorm.Scale);
        var bias = context.CheckArgumentType<IRType>(target, InstacneNorm.Bias);

        return (input, scale, bias) switch
        {
            (DistributedType a, DistributedType b, DistributedType c) => Visit(a, b, c, target),
            (TensorType a, TensorType b, TensorType c) => Visit(a, b, c, target),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, InstacneNorm target)
    {
        var inputType = context.GetArgumentType<IRType>(target, InstacneNorm.Input);
        var returnType = context.GetReturnType<IRType>();
        switch (inputType, returnType)
        {
            case (TensorType, TensorType):
                return new()
                {
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
                };

            case (DistributedType inputDistributedType, DistributedType):
                var scaleType = context.GetArgumentType<DistributedType>(target, InstacneNorm.Scale);
                var biasType = context.GetArgumentType<DistributedType>(target, InstacneNorm.Bias);
                var ring = GetRingReduceCommunicate(scaleType, new[] { 0, 1 }) + GetRingReduceCommunicate(biasType, new[] { 0, 1 });
                var reCompute = inputDistributedType.NdSBP.Select((sbp, i) => sbp is SBPSplit ? 1 : inputDistributedType.Placement.Hierarchy[i]).ToArray().Aggregate(1, (acc, rep) => acc * rep);
                return new()
                {
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + ring,
                    [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, 1) * (UInt128)reCompute,
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType) + ring,
                };
            default:
                throw new NotSupportedException();
        }
    }

    private IRType Visit(TensorType input, TensorType scale, TensorType bias, InstacneNorm target)
    {
        if (target.PackedAxes.Count == 0)
        {
            return input;
        }

        if (!(target.PackedAxes.Count == 1 && target.PackedAxes[0] == 1 && scale.DType is VectorType && bias.DType is VectorType))
        {
            return new InvalidType("when packed on channel, the scale and bias must be packed");
        }

        return input;
    }

    private IRType Visit(DistributedType input, DistributedType scale, DistributedType bias, InstacneNorm target)
    {
        var invalid = new InvalidType($"{input}, {scale}, {bias} not support");
        if (input.Placement != scale.Placement || scale.Placement != bias.Placement)
        {
            return invalid;
        }

        if (Visit(input.TensorType, scale.TensorType, bias.TensorType, target) is not TensorType tensorType)
        {
            return invalid;
        }

        var ndsbp = new SBP[input.Placement.Rank];

        for (int i = 0; i < input.Placement.Rank; i++)
        {
            switch (input.NdSBP[i], scale.NdSBP[i], bias.NdSBP[i])
            {
                case (SBPSplit { Axis: int ix }, SBPSplit { Axis: int sx }, SBPSplit { Axis: int bx }) when ix == 1 && sx == 0 && bx == 0:
                    ndsbp[i] = SBP.S(ix);
                    break;
                case (SBPSplit { Axis: int ix }, SBPBroadCast, SBPBroadCast) when ix != 1:
                    ndsbp[i] = SBP.S(ix);
                    break;
                case (SBPBroadCast, SBPBroadCast, SBPBroadCast):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }

    private UInt128 GetRingReduceCommunicate(DistributedType distributedType, int[] axes)
    {
        var ttype = Utilities.DistributedUtility.GetDividedTensorType(distributedType);
        var splits = axes.Where(i => i < distributedType.Placement.Rank && distributedType.NdSBP[i] is SBPSplit);
        if (!splits.Any())
        {
            return 0;
        }

        var p = (UInt128)splits.Select(i => distributedType.Placement.Hierarchy[i]).Aggregate(1, (acc, i) => acc * i);
        var v = CostUtility.GetMemoryAccess(distributedType.TensorType);
        return (p - 1) * (v / p);
    }
}
