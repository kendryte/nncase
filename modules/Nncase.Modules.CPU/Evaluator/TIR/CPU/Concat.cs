// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class ConcatEvaluator : ITypeInferencer<Concat>
{
    public IRType Visit(ITypeInferenceContext context, Concat target)
    {
        context.CheckArgumentType<TensorType>(target, Concat.Input);
        context.CheckArgumentType<TensorType>(target, Concat.Output);
        return TupleType.Void;
    }
}
