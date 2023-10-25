// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.XPU;

namespace Nncase.Evaluator.TIR.XPU;

public sealed class TDMAStoreEvaluator : ITypeInferencer<TDMAStore>
{
    public IRType Visit(ITypeInferenceContext context, TDMAStore target)
    {
        _ = context.CheckArgumentType<TensorType>(target, TDMAStore.Src);
        _ = context.CheckArgumentType<IRType>(target, TDMAStore.Dest);
        return TupleType.Void;
    }
}
