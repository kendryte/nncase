// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class TransposeEvaluator : ITypeInferencer<Transpose>
{
    public IRType Visit(ITypeInferenceContext context, Transpose target)
    {
        context.CheckArgumentType<TensorType>(target, Transpose.Input);
        context.CheckArgumentType<TensorType>(target, Transpose.Output);
        return TupleType.Void;
    }
}
