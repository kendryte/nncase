// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="RoPE"/>.
/// </summary>
public class RoPEEvaluator : IEvaluator<RoPE>, ITypeInferencer<RoPE>, ICostEvaluator<RoPE>,
    IMetricEvaluator<RoPE>
{
    public static bool AxisEqual(IRArray<SBP> a, IRArray<SBP> b, int startA, int startB)
    {
        var lenA = a.Count - startA;
        if (lenA != b.Count - startB)
        {
            return false;
        }

        for (int i = 0; i < lenA; i++)
        {
            if (!Equals(a[startA + i], b[startB + i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, RoPE target)
    {
        var inputTensor = context.GetArgumentValueAsTensor(target, RoPE.Input);
        var cosTensor = context.GetArgumentValueAsTensor(target, RoPE.Cos);
        var sinTensor = context.GetArgumentValueAsTensor(target, RoPE.Sin);

        var originDtype = inputTensor.ElementType;
        var legalOriginDtype = originDtype.Legalize([(DataTypes.Float16, DataTypes.Float32)]);
        inputTensor = inputTensor.CastElementTo(legalOriginDtype);
        cosTensor = cosTensor.CastElementTo(legalOriginDtype);
        sinTensor = sinTensor.CastElementTo(legalOriginDtype);

        var input = inputTensor.ToOrtTensor();
        var cos = cosTensor.ToOrtTensor();
        var sin = sinTensor.ToOrtTensor();

        var sliceAxis = inputTensor.Dimensions.Length - 1;
        var sliceDim = inputTensor.Dimensions[sliceAxis] / 2;
        var parts = OrtKI.Split(input, new[] { sliceDim, sliceDim }, sliceAxis);

        // rotate half
        var rotated = OrtKI.Concat([OrtKI.Neg(parts[1]), parts[0]], sliceAxis);
        var output = OrtKI.Add(OrtKI.Mul(input, cos), OrtKI.Mul(rotated, sin)).ToTensor(legalOriginDtype);
        return Value.FromTensor(output.CastElementTo(originDtype));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, RoPE target)
    {
        var input = context.CheckArgumentType<IRType>(target, RoPE.Input);
        var cos = context.CheckArgumentType<IRType>(target, RoPE.Cos);
        var sin = context.CheckArgumentType<IRType>(target, RoPE.Sin);

        return (input, cos, sin) switch
        {
            (DistributedType a, DistributedType b, DistributedType c) => Visit(a, b, c),
            (TensorType a, TensorType, TensorType) => Visit(a),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, RoPE target)
    {
        var inputType = context.GetArgumentType<IRType>(target, RoPE.Input);
        var cosType = context.GetArgumentType<IRType>(target, RoPE.Cos);
        var sinType = context.GetArgumentType<IRType>(target, RoPE.Sin);
        var macPerElement = 4; // 2 for mul, 1 for add, 1 for neg and concat
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(cosType) + CostUtility.GetMemoryAccess(sinType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, macPerElement),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, RoPE target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, RoPE.Input);
        var cosType = context.GetArgumentType<TensorType>(target, RoPE.Cos);
        var sinType = context.GetArgumentType<TensorType>(target, RoPE.Sin);
        var returnType = context.GetReturnType<TensorType>();
        var macPerElement = 4; // 2 for mul, 1 for add, 1 for neg and concat

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(cosType) + CostUtility.GetMemoryAccess(sinType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = CostUtility.GetCPUCycles(returnType, macPerElement),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }

    private IRType Visit(DistributedType input, DistributedType scale, DistributedType bias)
    {
        // only unsupported print without to-string
        if (input.Placement != scale.Placement || scale.Placement != bias.Placement
            || !AxisEqual(input.AxisPolicies, scale.AxisPolicies, startA: 1, startB: 0)
            || !AxisEqual(scale.AxisPolicies, bias.AxisPolicies, startA: 0, startB: 0)
            || input.AxisPolicies[^1] is not SBPBroadCast)
        {
            return new InvalidType("RoPE: distributed types mismatch (placement/axis/SBP)");

            // optional(still ToString)：
            // return new InvalidType($"RoPE mismatch: in={input.GetType().Name}, cos={scale.GetType().Name}, sin={bias.GetType().Name}");
        }

        return input;
    }
}
