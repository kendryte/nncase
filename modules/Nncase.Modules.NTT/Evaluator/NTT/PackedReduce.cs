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

public sealed class PackedReduceEvaluator : IEvaluator<PackedReduce>, ITypeInferencer<PackedReduce>, ICostEvaluator<PackedReduce>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, PackedReduce target)
    {
        var input = context.GetOrtArgumentValue(target, PackedReduce.Input);
        var padedNums = context.GetArgumentValueAsArray<int>(target, PackedReduce.PadedNums);
        var inshape = input.Shape.SkipLast(target.PackedAxes.Count).Select(i => i).ToArray();
        var inlanes = input.Shape.TakeLast(target.PackedAxes.Count).Select(i => (int)i).ToArray();
        var unpackedInput = NTTEvaluatorUtility.UnpackTensor(input, target.PackedAxes, padedNums, out _);
        var axes = target.Axes.Select(i => (long)i).ToArray();
        long keepdims = target.KeepDims ? 1 : 0;
        foreach (var axis in target.PackedAxes.Reverse())
        {
            input = input.Unpack(axis);
        }

        OrtKISharp.Tensor output;
        switch (target.ReduceOp)
        {
            case ReduceOp.Sum:
                output = OrtKI.ReduceSum(unpackedInput, axes, keepdims, 0);
                break;
            case ReduceOp.Mean:
                output = OrtKI.ReduceMean(unpackedInput, axes, keepdims);
                break;
            default:
                throw new NotSupportedException(target.ReduceOp.ToString());
        }

        var (outPackAxes, outPadNums, outLanes, outShape) = PackedReduce.ComputeOutputInfo(target, padedNums.Select(p => (Dimension)p).ToArray(), inshape, inlanes);
        output = NTTEvaluatorUtility.RepackTensor(output, outLanes.ToArray(), outPackAxes, outPadNums.Select(x => (int)x.FixedValue).ToArray());

        return Value.FromTensor(Tensor.FromBytes(outLanes.Length == 0 ? DataTypes.Float32 : new VectorType(DataTypes.Float32, outLanes.ToArray()), output.BytesBuffer.ToArray(), outShape));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, PackedReduce target)
    {
        var input = context.CheckArgumentType<IRType>(target, PackedReduce.Input);

        return input switch
        {
            DistributedType d => Visit(context, target, d),
            TensorType t => Visit(context, target, t),
            AnyType a => a,
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, PackedReduce target)
    {
        var input = context.GetArgumentType<IRType>(target, PackedReduce.Input);
        var ret = context.GetReturnType<IRType>();
        var inputShape = input switch
        {
            TensorType t => t.Shape,
            DistributedType d => d.TensorType.Shape,
            _ => throw new NotSupportedException(string.Empty),
        };
        var retShape = ret switch
        {
            TensorType t => t.Shape,
            DistributedType d => d.TensorType.Shape,
            _ => throw new NotSupportedException(string.Empty),
        };
        uint input_elem = inputShape.Aggregate(1U, (acc, d) => acc * (d.IsFixed ? (uint)d.FixedValue : 1U));
        uint ret_elem = retShape.Aggregate(1U, (acc, d) => acc * (d.IsFixed ? (uint)d.FixedValue : 1U));
        uint macPerElement = input_elem / ret_elem;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, macPerElement),
        };
    }

    private IRType Visit(ITypeInferenceContext context, PackedReduce target, TensorType t)
    {
        var inshape = (RankedShape)t.Shape;
        var inDtype = (VectorType)t.DType;
        var inlanes = inDtype.Lanes.ToArray();
        var paddedNums = ((RankedShape)context.GetArgument(target, PackedReduce.PadedNums)).Dimensions.ToArray();
        var (_, _, outLanes, outShape) = PackedReduce.ComputeOutputInfo(target, paddedNums, inshape, inlanes);
        var outDType = outLanes.Length == 0 ? inDtype.ElemType : new VectorType(inDtype.ElemType, outLanes);
        return new TensorType(outDType, outShape);
    }

    private IRType Visit(ITypeInferenceContext context, PackedReduce target, DistributedType input)
    {
        if (Visit(context, target, input.TensorType) is not TensorType tensorType)
        {
            throw new InvalidOperationException();
        }

        var axes = target.Axes.ToArray();
        var ndsbp = new SBP[input.TensorType.Shape.Rank];

        for (int i = 0; i < ndsbp.Length; i++)
        {
            switch (input.AxisPolicies[i])
            {
                case SBPSplit split when axes.Contains(i):
                    // ndsbp[i] = SBP.P(target.ReduceOp);
                    return new InvalidType("reduce not support split on axes for now.");
                default:
                    ndsbp[i] = input.AxisPolicies[i];
                    break;
            }
        }

        return new DistributedType(tensorType, tensorType.Shape.Rank == ndsbp.Length ? ndsbp : ndsbp.Where((_, i) => !axes.Contains(i)).ToArray(), input.Placement);
    }
}
