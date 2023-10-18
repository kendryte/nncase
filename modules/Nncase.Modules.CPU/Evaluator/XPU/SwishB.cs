// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.XPU;

namespace Nncase;

public sealed class SwishBEvaluator : ITypeInferencer<SwishB>
{
    public IRType Visit(ITypeInferenceContext context, SwishB target)
    {
        context.CheckArgumentType<TensorType>(target, SwishB.Input);
        context.CheckArgumentType<TensorType>(target, SwishB.Output);
        return TupleType.Void;
    }
}
