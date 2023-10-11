// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.XPU;

namespace Nncase.Evaluator.XPU;

public sealed class ReduceArgEvaluator : ITypeInferencer<ReduceArg>
{
    public IRType Visit(ITypeInferenceContext context, ReduceArg target)
    {
        return TupleType.Void;
    }
}
