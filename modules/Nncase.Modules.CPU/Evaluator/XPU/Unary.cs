// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.XPU;

namespace Nncase;

public sealed class UnaryEvaluator : ITypeInferencer<Unary>
{
    public IRType Visit(ITypeInferenceContext context, Unary target)
    {
        context.CheckArgumentType<TensorType>(target, Unary.Input);
        context.CheckArgumentType<TensorType>(target, Unary.Output);
        return TupleType.Void;
    }
}
