// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;

namespace Nncase.Evaluator.CPU;

/// <summary>
/// Evaluator for <see cref="CPUKernelOp"/>.
/// </summary>
public class CPUKernelOpEvaluator : IEvaluator<CPUKernelOp>, ITypeInferencer<CPUKernelOp>, ICostEvaluator<CPUKernelOp>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, CPUKernelOp target)
    {
        return CompilerServices.EvaluateOp(target.Target, context);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, CPUKernelOp target)
    {
        return CompilerServices.InferenceOp(target.Target, context, new());
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, CPUKernelOp target)
    {
        return CompilerServices.EvaluateOpCost(target.Target, context);
    }
}
