// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.Intrinsics.X86;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="RangeOf"/>.
/// </summary>
public class RangeOfEvaluator : IEvaluator<RangeOf>, ITypeInferencer<RangeOf>, ICostEvaluator<RangeOf>, IShapeEvaluator<RangeOf>, IMetricEvaluator<RangeOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, RangeOf target)
    {
        var input = context.GetArgumentValueAsTensor<float>(target, RangeOf.Input);
        var min = float.MaxValue;
        var max = float.MinValue;
        foreach (var f in input.Buffer.Span)
        {
            if (float.IsFinite(f))
            {
                min = System.Math.Min(min, f);
                max = System.Math.Max(max, f);
            }
        }

        if (target.IsMatmulArg0)
        {
            var k = context.CurrentCall.Arguments[0].CheckedShape.ToValueArray().Last();
            var m = context.CurrentCall.Arguments[0].CheckedShape.Prod().Value! / k;

            var rangeByK = new float[k * 2];
            for (int i = 0; i < k; i++)
            {
                var minM = float.MaxValue;
                var maxM = float.MinValue;
                for (int j = 0; j < m; j++)
                {
                    var value = input.Buffer.Span[(j * k) + i];
                    if (float.IsFinite(value))
                    {
                        minM = System.Math.Min(minM, value);
                        maxM = System.Math.Max(maxM, value);
                    }
                }

                rangeByK[(i * 2) + 0] = minM;
                rangeByK[(i * 2) + 1] = maxM;
            }

            var rangeByM = new float[(int)(m * 2)];
            for (int i = 0; i < m; i++)
            {
                var minK = float.MaxValue;
                var maxK = float.MinValue;
                for (int j = 0; j < k; j++)
                {
                    var value = input.Buffer.Span[(i * k) + j];
                    if (float.IsFinite(value))
                    {
                        minK = System.Math.Min(minK, value);
                        maxK = System.Math.Max(maxK, value);
                    }
                }

                rangeByM[(i * 2) + 0] = minK;
                rangeByM[(i * 2) + 1] = maxK;
            }

            return new TupleValue(new IValue[] { Value.FromTensor(new[] { min, max }), Value.FromTensor(Tensor.FromArray(rangeByK)), Value.FromTensor(Tensor.FromArray(rangeByM)) });
        }

        return Value.FromTensor(new[] { min, max });
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, RangeOf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, RangeOf.Input);
        if (target.IsMatmulArg0)
        {
            var k = context.GetArgument(target, RangeOf.Input).CheckedShape.ToValueArray().Last();
            var m = context.GetArgument(target, RangeOf.Input).CheckedShape.Prod().Value! / k;
            return new TupleType(new[] { input with { Shape = new Shape(2) }, input with { Shape = new Shape(k * 2) }, input with { Shape = new Shape((int)m * 2) } });
        }

        return input with { Shape = new Shape(2) };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, RangeOf target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, RangeOf.Input);
        IRType outputType;
        if (target.IsMatmulArg0)
        {
            outputType = context.GetReturnType<TupleType>().ToArray()[0];
        }
        else
        {
            outputType = context.GetReturnType<TensorType>();
        }

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, 2),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, RangeOf target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, RangeOf.Input);
        _ = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType, 2),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, RangeOf target) => context.GetArgumentShape(target, RangeOf.Input);
}
