// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.XPU;

namespace Nncase.Evaluator.TIR.XPU;

public class TDMALoadEvaluator : ITypeInferencer<TDMALoad>
{
    public IRType Visit(ITypeInferenceContext context, TDMALoad target)
    {
        _ = context.CheckArgumentType<TensorType>(target, TDMALoad.Dest);
        _ = context.CheckArgumentType<IRType>(target, TDMALoad.Src);
        return TupleType.Void;
    }
}
