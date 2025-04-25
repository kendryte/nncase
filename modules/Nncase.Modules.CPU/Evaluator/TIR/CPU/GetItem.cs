// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class GetItemEvaluator : ITypeInferencer<GetItem>
{
    public IRType Visit(ITypeInferenceContext context, GetItem target)
    {
        context.CheckArgumentType<TensorType>(target, GetItem.Input);
        context.CheckArgumentType<TensorType>(target, GetItem.Index);
        return TupleType.Void;
    }
}
