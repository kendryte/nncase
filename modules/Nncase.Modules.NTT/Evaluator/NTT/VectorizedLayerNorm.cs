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

public sealed class VectorizedLayerNormEvaluator : IEvaluator<VectorizedLayerNorm>, ITypeInferencer<VectorizedLayerNorm>, ICostEvaluator<VectorizedLayerNorm>,
    IMetricEvaluator<VectorizedLayerNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, VectorizedLayerNorm target)
    {
        var input = context.GetOrtArgumentValue(target, VectorizedLayerNorm.Input);
        var scale = context.GetOrtArgumentValue(target, VectorizedLayerNorm.Scale);
        var bias = context.GetOrtArgumentValue(target, VectorizedLayerNorm.Bias);
        var padedNums = context.GetArgumentValueAsArray<int>(target, VectorizedLayerNorm.PadedNums);
        var lanes = input.Shape.TakeLast(target.VectorizedAxes.Count).Select(i => (int)i).ToArray();
        var devectorizedInput = NTTEvaluatorUtility.DevectorizeTensor(input, target.VectorizedAxes, padedNums, out _);
        var vectorizeAxes = target.VectorizedAxes.Where(axis => axis >= target.Axis).Select(axis => axis - target.Axis).ToArray();
        var pPadedNums = padedNums.Skip(target.VectorizedAxes.Count - vectorizeAxes.Length).ToArray();
        var devectorizedScale = NTTEvaluatorUtility.DevectorizeTensor(scale, vectorizeAxes, pPadedNums, out _);
        var devectorizedBias = NTTEvaluatorUtility.DevectorizeTensor(bias, vectorizeAxes, pPadedNums, out _);

        var shape = devectorizedInput.Shape;
        var inputBuffer = devectorizedInput.BytesBuffer.ToArray();
        var inputSpan = MemoryMarshal.Cast<byte, float>(inputBuffer);
        var scaleBuffer = devectorizedScale.BytesBuffer.ToArray();
        var scaleSpan = MemoryMarshal.Cast<byte, float>(scaleBuffer);
        var biasBuffer = devectorizedBias.BytesBuffer.ToArray();
        var biasSpan = MemoryMarshal.Cast<byte, float>(biasBuffer);

        var output = NN.LayerNormEvaluator.LayerNormImpl(shape, inputSpan, scaleSpan, biasSpan, target.Axis, target.Epsilon, target.UseMean);
        var outputTensor = OrtKISharp.Tensor.MakeTensor(new Memory<float>(output), OrtDataType.Float, devectorizedInput.Shape);
        outputTensor = NTTEvaluatorUtility.RevectorizeTensor(outputTensor, lanes, target.VectorizedAxes, padedNums);

        return Value.FromTensor(Tensor.FromBytes(new VectorType(DataTypes.Float32, lanes), outputTensor.BytesBuffer.ToArray(), outputTensor.Shape.SkipLast(target.VectorizedAxes.Count).Select(i => i).ToArray()));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, VectorizedLayerNorm target)
    {
        var input = context.CheckArgumentType<IRType>(target, VectorizedLayerNorm.Input);
        var scale = context.CheckArgumentType<IRType>(target, VectorizedLayerNorm.Scale);
        var bias = context.CheckArgumentType<IRType>(target, VectorizedLayerNorm.Bias);

        return (input, scale, bias) switch
        {
            (DistributedType a, DistributedType b, DistributedType c) => Visit(a, b, c, target.Axis),
            (TensorType a, TensorType, TensorType) => Visit(a),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, VectorizedLayerNorm target)
    {
        var inputType = context.GetArgumentType<IRType>(target, VectorizedLayerNorm.Input);
        var returnType = context.GetReturnType<IRType>();
        switch (inputType, returnType)
        {
            case (TensorType, TensorType):
                return new()
                {
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
                    [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, 1),
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
                };

            case (DistributedType, DistributedType):
                var scaleType = context.GetArgumentType<DistributedType>(target, VectorizedLayerNorm.Scale);
                var biasType = context.GetArgumentType<DistributedType>(target, VectorizedLayerNorm.Bias);

                // var ring = GetRingReduceCommunicate(scaleType, new[] { 0, 1 }) + GetRingReduceCommunicate(biasType, new[] { 0, 1 });
                // var reCompute = inputDistributedType.NdSBP.Select((sbp, i) => sbp is SBPSplit ? 1 : inputDistributedType.Placement.Hierarchy[i]).ToArray().Aggregate(1, (acc, rep) => acc * rep);
                return new()
                {
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(scaleType) + CostUtility.GetMemoryAccess(biasType),
                    [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, 1),
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
                };
            default:
                throw new NotSupportedException();
        }
    }

    public Metric Visit(IMetricEvaluateContext context, VectorizedLayerNorm target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, VectorizedLayerNorm.Input);
        var returnType = context.GetReturnType<TensorType>();

        var r = MetricUtility.GetFLOPs(returnType);
        var i = MetricUtility.GetFLOPs(inputType);
        var outter = i / r;
        var inner = i / outter;

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = outter * ((inner * 7) + MetricUtility.SqrtFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }

    private IRType Visit(DistributedType input, DistributedType scale, DistributedType bias, int raxis)
    {
        var invalid = new InvalidType($"{input}, {scale}, {bias} not support");
        if (input.Placement != scale.Placement || scale.Placement != bias.Placement)
        {
            return invalid;
        }

        var ndsbp = new SBP[input.AxisPolicies.Count];

        for (int i = 0; i < ndsbp.Length; i++)
        {
            var scalePolicy = i - raxis >= 0 ? scale.AxisPolicies[i - raxis] : null;
            var biasPolicy = i - raxis >= 0 ? bias.AxisPolicies[i - raxis] : null;
            switch (input.AxisPolicies[i], scalePolicy, biasPolicy)
            {
                case (SBPSplit si, SBPSplit ss, SBPSplit sb) when i >= raxis && si.Axes == ss.Axes && ss.Axes == sb.Axes:
                    // FIXME: not support on axes for now
#if true
                    return invalid;
#else
                    ndsbp[i] = si;
                    break;
#endif
                case (SBPSplit si, _, _) when i < raxis:
                    ndsbp[i] = si;
                    break;
                case (SBPBroadCast, SBPBroadCast or null, SBPBroadCast or null):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(input.TensorType, ndsbp, input.Placement);
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
