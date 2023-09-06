// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.CPU;

namespace Nncase.Evaluator.CPU;

public class TDMALoadEvaluator : ITypeInferencer<TDMALoad>
{
    public IRType Visit(ITypeInferenceContext context, TDMALoad target)
    {
        var inType = context.CheckArgumentType<TensorType>(target, TDMALoad.Input);
        var outType = context.CheckArgumentType<TensorType>(target, TDMALoad.Output);
        return TupleType.Void;
    }
}
