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
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Stack"/>.
/// </summary>
public class StackEvaluator : IEvaluator<Stack>, ITypeInferencer<Stack>, ICostEvaluator<Stack>
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
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, 1),
        };
    }

    private IRType Visit(ITypeInferenceContext context, Stack target, TupleType inputs)
    {
        if (context.GetArgument(target, Stack.Axis) is TensorConst axis_con)
        {
            var axis_v = axis_con.Value.ToScalar<int>();
            var ttypes = new TensorType[inputs.Count];
            foreach (var (i, input) in Enumerable.Range(0, inputs.Count).Zip(inputs))
            {
                if (input is TensorType ttype)
                {
                    if (ttype.Shape.IsUnranked)
                    {
                        return ttype;
                    }

                    ttypes[i] = ttype;
                }
                else
                {
                    return new InvalidType("The Tuple Elements Type Must All Equals TensorType");
                }
            }

            if (!ttypes.Skip(1).All(ttype => ttype.Shape == ttypes[0].Shape))
            {
                return new InvalidType("The Tuple Elements Shape Must All Equal!");
            }

            if (ttypes[0].Shape.IsScalar)
            {
                if (axis_v != 0)
                {
                    return new InvalidType("Axis must be zero when stack scalar");
                }

                return ttypes[0] with { Shape = new Shape(inputs.Count) };
            }
            else
            {
                var outshape = ttypes[0].Shape.ToList();
                outshape.Insert(axis_v, inputs.Count);
                return ttypes[0] with { Shape = new Shape(outshape) };
            }
        }

        return new InvalidType("The Stack Axis Must Be Const!");
    }
}
