// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.XPU;

namespace Nncase.Evaluator.XPU;

public class TDMALoadEvaluator : ITypeInferencer<TDMALoad>
{
    public IRType Visit(ITypeInferenceContext context, TDMALoad target)
    {
        var inType = context.CheckArgumentType<TensorType>(target, TDMALoad.Dest);
        var outType = context.CheckArgumentType<TensorType>(target, TDMALoad.Src);
        return TupleType.Void;
    }
}
