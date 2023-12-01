// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public class TensorLoadEvaluator : ITypeInferencer<TensorLoad>
{
    public IRType Visit(ITypeInferenceContext context, TensorLoad target)
    {
        _ = context.CheckArgumentType<TensorType>(target, TensorLoad.Dest);
        _ = context.CheckArgumentType<IRType>(target, TensorLoad.Src);
        return TupleType.Void;
    }
}
