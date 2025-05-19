// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class StackEvaluator : ITypeInferencer<Stack>
{
    public IRType Visit(ITypeInferenceContext context, Stack target)
    {
        context.CheckArgumentType<TensorType>(target, Stack.Input);
        context.CheckArgumentType<TensorType>(target, Stack.Output);
        return TupleType.Void;
    }
}
