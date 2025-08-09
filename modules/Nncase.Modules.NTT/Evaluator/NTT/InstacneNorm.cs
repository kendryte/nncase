// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.IR.NTT;

public sealed class InstanceNormEvaluator : IEvaluator<InstacneNorm>, ITypeInferencer<InstacneNorm>, ICostEvaluator<InstacneNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, InstacneNorm target)
    {
        var input = context.GetOrtArgumentValue(target, InstacneNorm.Input);
        var scale = context.GetOrtArgumentValue(target, InstacneNorm.Scale);
        var bias = context.GetOrtArgumentValue(target, InstacneNorm.Bias);
        var padedNums = context.GetArgumentValueAsArray<int>(target, InstacneNorm.PadedNums);
        if (target.VectorizedAxes.Count == 0)
        {
            return Value.FromTensor(OrtKI.InstanceNormalization(input, scale, bias, target.Epsilon).ToTensor());
        }
        else
        {
            var lanes = input.Shape.TakeLast(target.VectorizedAxes.Count).Select(i => (int)i).ToArray();
            var channelPadNums = 0;
            for (int i = 0; i < target.VectorizedAxes.Count; i++)
            {
                var axis = target.VectorizedAxes[i];
                if (axis == 1)
                {
                    channelPadNums = padedNums[i];
                }

                input = input.Unpack(target.VectorizedAxes.Count - i, axis);
            }

            if (channelPadNums > 0)
            {
                var rk = target.VectorizedAxes.Count;
                var inshape = input.Shape.ToArray();
                input = OrtKI.Slice(
                    input,
                    target.VectorizedAxes.Select(_ => 0L).ToArray(),
                    Enumerable.Range(0, rk).Select(i => inshape[target.VectorizedAxes[i]] - padedNums[i]).ToArray(),
                    target.VectorizedAxes.Select(axis => (long)axis).ToArray(),
                    target.VectorizedAxes.Select(_ => 1L).ToArray());
            }

            if (scale.Shape.Length == 2)
            {
                scale = scale.Unpack(1, 0);
                if (channelPadNums > 0)
                {
                    scale = OrtKI.Slice(scale, new[] { 0L }, new[] { scale.Shape[0] - channelPadNums }, new[] { 0L }, new[] { 1L });
                }
            }

            if (bias.Shape.Length == 2)
            {
                bias = bias.Unpack(1, 0);
                if (channelPadNums > 0)
                {
                    bias = OrtKI.Slice(bias, new[] { 0L }, new[] { bias.Shape[0] - channelPadNums }, new[] { 0L }, new[] { 1L });
                }
            }

            var norm = OrtKI.InstanceNormalization(input, scale, bias, target.Epsilon);
            var output = NTTEvaluatorUtility.RevectorizeTensor(norm, lanes, target.VectorizedAxes, padedNums);
            return Value.FromTensor(Tensor.FromBytes(new TensorType(new VectorType(norm.DataType.ToDataType(), lanes), output.Shape.SkipLast(target.VectorizedAxes.Count).Select(i => (int)i).ToArray()), output.BytesBuffer.ToArray()));
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

#if false
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
#endif
            default:
                throw new NotSupportedException();
        }
    }

    private IRType Visit(TensorType input, TensorType scale, TensorType bias, InstacneNorm target)
    {
        if (target.VectorizedAxes.Count == 0)
        {
            return input;
        }

        if (!(target.VectorizedAxes.Count == 1 && target.VectorizedAxes[0] == 1 && scale.DType is VectorType && bias.DType is VectorType))
        {
            return new InvalidType("when vectorized on channel, the scale and bias must be vectorized");
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

        var ndsbp = new SBP[input.AxisPolicies.Count];
        var rAxis = 1;
        for (int i = 0; i < ndsbp.Length; i++)
        {
            var scalePolicy = i - rAxis == 0 ? scale.AxisPolicies[i - rAxis] : null;
            var biasPolicy = i - rAxis == 0 ? bias.AxisPolicies[i - rAxis] : null;
            switch (input.AxisPolicies[i], scalePolicy, biasPolicy)
            {
                case (SBPSplit si, SBPSplit ss, SBPSplit sb) when i == rAxis && si.Axes == ss.Axes && ss.Axes == sb.Axes:
                    ndsbp[i] = si;
                    break;
                case (SBPSplit si, _, _) when i != rAxis:
                    ndsbp[i] = si;
                    break;
                case (SBPBroadCast, SBPBroadCast or null, SBPBroadCast or null):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }

#if false
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
#endif
}
