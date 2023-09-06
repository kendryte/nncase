// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.


using Nncase.IR;

namespace Nncase.Evaluator.CPU;

public sealed class TDMAStoreEvaluator : ITypeInferencer<TDMAStore>
{
    public IRType Visit(ITypeInferenceContext context, TDMAStore target)
    {
        var inType = context.CheckArgumentType<TensorType>(target, TDMAStore.Input);
        var outType = context.CheckArgumentType<TensorType>(target, TDMAStore.Output);
        return TupleType.Void;
    }
}
