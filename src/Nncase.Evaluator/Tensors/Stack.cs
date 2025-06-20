// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Stack"/>.
/// </summary>
public class StackEvaluator : IEvaluator<Stack>, ITypeInferencer<Stack>, ICostEvaluator<Stack>, IMetricEvaluator<Stack>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Stack stack)
    {
        var inputs = context.GetArgumentValueAsTensors(stack, Stack.Inputs);
        var axis = context.GetArgumentValueAsScalar<long>(stack, Stack.Axis);
        var ort_inputs = inputs.Select(t =>
        {
            var ort = t.ToOrtTensor();
            var old_shape = ort.Shape;
            var new_shape = old_shape.Take((int)axis).Concat(new long[] { 1 }).Concat(old_shape.Skip((int)axis)).ToArray();
            ort.Reshape(new_shape);
            return ort;
        }).ToArray();
        return OrtKI.Concat(ort_inputs, axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Stack target)
    {
        var inputs = context.CheckArgumentType<TupleType>(target, Stack.Inputs);
        return Visit(context, target, inputs);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Stack target)
    {
        var input = context.GetArgumentType<TupleType>(target, Stack.Inputs);
        var ret = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, 1),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Stack target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Stack target, TupleType inputs)
    {
        if (inputs.Count == 0)
        {
            return new InvalidType("Tuple count should not be zero");
        }

        if (context.GetArgument(target, Stack.Axis) is DimConst axis_con)
        {
            var axis_v = axis_con.Value;
            var firstType = inputs[0];
            if (inputs.Any(x => x != firstType))
            {
                return new InvalidType("The Tuple Elements Type Must All Equal!");
            }

            var tensorType = firstType switch
            {
                TensorType t => t,
                DistributedType dt => dt.TensorType,
                _ => throw new TypeInferenceInterruptException(new InvalidType("The Tuple Elements Must Be TensorType!")),
            };

            var axisPolices = firstType switch
            {
                TensorType t => Array.Empty<SBP>(),
                DistributedType dt => dt.AxisPolicies,
                _ => throw new TypeInferenceInterruptException(new InvalidType("The Tuple Elements Must Be TensorType!")),
            };

            if (tensorType.IsScalar)
            {
                if (axis_v != 0)
                {
                    return new InvalidType("Axis must be zero when stack scalar");
                }

                tensorType = tensorType with { Shape = new RankedShape(inputs.Count) };
            }
            else if (tensorType.Shape is RankedShape inShape)
            {
                var outshape = inShape.ToList();
                outshape.Insert((int)(axis_v < 0 ? inShape.Count + axis_v : axis_v), inputs.Count);
                tensorType = tensorType with { Shape = new RankedShape(outshape) };
            }
            else
            {
                tensorType = tensorType with { Shape = Shape.Unranked };
            }

            return firstType is DistributedType dt2 ? dt2 with { TensorType = tensorType, AxisPolicies = axisPolices } : tensorType;
        }

        return new InvalidType("The Stack Axis Must Be Const!");
    }
}
