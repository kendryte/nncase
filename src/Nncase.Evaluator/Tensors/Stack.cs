// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Stack"/>.
/// </summary>
public class StackEvaluator : IEvaluator<Stack>, ITypeInferencer<Stack>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Stack stack)
    {
        var inputs = context.GetArgumentExpr(stack, Stack.Inputs);
        var axis = context.GetArgumentValueAsScalar<long>(stack, Stack.Axis);
        var inputTensors = ((IR.Tuple)inputs).Select(x => context.GetOrtValue(x)).ToArray();
        return OrtKI.ConcatFromSequence(inputTensors, axis, inputTensors.Length).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Stack target)
    {
        var inputs = context.CheckArgumentType<TupleType>(target, Stack.Inputs);
        return Visit(context, target, inputs);
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
