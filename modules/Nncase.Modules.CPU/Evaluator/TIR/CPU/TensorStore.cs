// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class TensorStoreEvaluator : ITypeInferencer<TensorStore>
{
    public IRType Visit(ITypeInferenceContext context, TensorStore target)
    {
        _ = context.CheckArgumentType<TensorType>(target, TensorStore.Src);
        _ = context.CheckArgumentType<IRType>(target, TensorStore.Dest);
        return TupleType.Void;
    }
}
