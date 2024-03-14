// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class SliceEvaluator : ITypeInferencer<Slice>
{
    public IRType Visit(ITypeInferenceContext context, Slice target)
    {
        context.CheckArgumentType<TensorType>(target, Slice.Input);
        context.CheckArgumentType<TensorType>(target, Slice.Output);
        return TupleType.Void;
    }
}
