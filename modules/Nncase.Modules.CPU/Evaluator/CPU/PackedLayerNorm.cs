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

public sealed class PackedLayerNormEvaluator : IEvaluator<PackedLayerNorm>, ITypeInferencer<PackedLayerNorm>, ICostEvaluator<PackedLayerNorm>,
    IShapeEvaluator<PackedLayerNorm>, IMetricEvaluator<PackedLayerNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, PackedLayerNorm target)
    {
        var input = context.GetOrtArgumentValue(target, PackedLayerNorm.Input);
        var scale = context.GetOrtArgumentValue(target, PackedLayerNorm.Scale);
        var bias = context.GetOrtArgumentValue(target, PackedLayerNorm.Bias);
        var lanes = input.Shape.TakeLast(target.PackedAxes.Count).Select(i => (int)i).ToArray();
        var unpackedInput = CPUEvaluatorUtility.UnpackTensor(input, target.PackedAxes, target.PadedNums, out _);
        var packAxes = target.PackedAxes.Where(axis => axis >= target.Axis).Select(axis => axis - target.Axis).ToArray();
        var padedNums = target.PadedNums.Skip(target.PackedAxes.Count - packAxes.Length).ToArray();
        var unpackedScale = CPUEvaluatorUtility.UnpackTensor(scale, packAxes, padedNums, out _);
        var unpackedBias = CPUEvaluatorUtility.UnpackTensor(bias, packAxes, padedNums, out _);

        var shape = unpackedInput.Shape.Select(i => (int)i).ToArray();
        var inputBuffer = unpackedInput.BytesBuffer.ToArray();
        var inputSpan = MemoryMarshal.Cast<byte, float>(inputBuffer);
        var scaleBuffer = unpackedScale.BytesBuffer.ToArray();
        var scaleSpan = MemoryMarshal.Cast<byte, float>(scaleBuffer);
        var biasBuffer = unpackedBias.BytesBuffer.ToArray();
        var biasSpan = MemoryMarshal.Cast<byte, float>(biasBuffer);

        var output = NN.LayerNormEvaluator.LayerNormImpl(shape, inputSpan, scaleSpan, biasSpan, target.Axis, target.Epsilon, target.UseMean);
        var outputTensor = OrtKISharp.Tensor.MakeTensor(new Memory<float>(output), OrtDataType.Float, unpackedInput.Shape);
        outputTensor = CPUEvaluatorUtility.RepackTensor(outputTensor, lanes, target.PackedAxes, target.PadedNums);

        return Value.FromTensor(Tensor.FromBytes(new VectorType(DataTypes.Float32, lanes), outputTensor.BytesBuffer.ToArray(), outputTensor.Shape.SkipLast(target.PackedAxes.Count).Select(i => (int)i).ToArray()));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, PackedLayerNorm target)
    {
        var input = context.CheckArgumentType<IRType>(target, PackedLayerNorm.Input);
        var scale = context.CheckArgumentType<IRType>(target, PackedLayerNorm.Scale);
        var bias = context.CheckArgumentType<IRType>(target, PackedLayerNorm.Bias);

        return (input, scale, bias) switch
        {
            (DistributedType a, DistributedType b, DistributedType c) => Visit(a, b, c, target.Axis),
            (TensorType a, TensorType, TensorType) => Visit(a),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, PackedLayerNorm target)
    {
        var inputType = context.GetArgumentType<IRType>(target, PackedLayerNorm.Input);
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
                var scaleType = context.GetArgumentType<DistributedType>(target, PackedLayerNorm.Scale);
                var biasType = context.GetArgumentType<DistributedType>(target, PackedLayerNorm.Bias);
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

    public Metric Visit(IMetricEvaluateContext context, PackedLayerNorm target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, PackedLayerNorm.Input);
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

    public Expr Visit(IShapeEvaluateContext context, PackedLayerNorm target) => context.GetArgumentShape(target, PackedLayerNorm.Input);

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

        var ndsbp = new SBP[input.Placement.Rank];

        for (int i = 0; i < input.Placement.Rank; i++)
        {
            switch (input.NdSBP[i], scale.NdSBP[i], bias.NdSBP[i])
            {
                case (SBPSplit { Axis: int ix }, SBPSplit { Axis: int sx }, SBPSplit { Axis: int bx }) when ix >= raxis && sx == (ix - raxis) && bx == sx:
                    ndsbp[i] = SBP.S(ix);
                    break;
                case (SBPSplit { Axis: int ix }, SBPBroadCast, SBPBroadCast) when ix < raxis:
                    ndsbp[i] = SBP.S(ix);
                    break;
                case (SBPBroadCast, SBPBroadCast, SBPBroadCast):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(input.TensorType, ndsbp, input.Placement);
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
