// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator.TIR;

public class MemcopyEvaluator : ITypeInferencer<Memcopy>
{
    public IRType Visit(ITypeInferenceContext context, Memcopy target)
    {
        _ = context.CheckArgumentType<TensorType>(target, Memcopy.Dest);
        _ = context.CheckArgumentType<TensorType>(target, Memcopy.Src);
        return TupleType.Void;
    }
}
