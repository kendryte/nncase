// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class SwishEvaluator : ITypeInferencer<Swish>
{
    public IRType Visit(ITypeInferenceContext context, Swish target)
    {
        context.CheckArgumentType<TensorType>(target, Swish.Input);
        context.CheckArgumentType<TensorType>(target, Swish.Output);
        return TupleType.Void;
    }
}
