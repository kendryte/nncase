// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CustomCPU;
using OrtKISharp;

namespace Nncase.Evaluator.CustomCPU;

public class UnaryEvaluator : IEvaluator<Unary>, ITypeInferencer<Unary>, ICostEvaluator<Unary>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unary unary)
    {
        var input_tensor = context.GetArgumentValueAsTensor(unary, Unary.Input);
        return Math.UnaryEvaluator.InferValue(input_tensor, unary.UnaryOp);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unary target)
    {
        var inputType = context.GetArgumentType(target, Unary.Input);
        if (CheckCustomSBP(inputType, target))
        {
            return Math.UnaryEvaluator.InferType(inputType, target.UnaryOp);
        }
        else
        {
            return new InvalidType("Not Match With CustomSBP");
        }
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Unary target)
    {
        return target.Cost;
    }

    private bool CheckCustomSBP(IRType input, Unary target)
    {
        if (input is DistributedType inType)
        {
            if (Enumerable.Range(0, inType.TensorType.Shape.Rank).Any(i => inType.AxisPolices[i] != target.InSBPs[i]))
            {
                return false;
            }
        }

        return true;
    }
}
