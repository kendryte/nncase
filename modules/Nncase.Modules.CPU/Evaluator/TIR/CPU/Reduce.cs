// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class ReduceEvaluator : ITypeInferencer<Reduce>
{
    public IRType Visit(ITypeInferenceContext context, Reduce target)
    {
        context.CheckArgumentType<TensorType>(target, Reduce.Input);
        context.CheckArgumentType<TensorType>(target, Reduce.Output);
        return TupleType.Void;
    }
}
