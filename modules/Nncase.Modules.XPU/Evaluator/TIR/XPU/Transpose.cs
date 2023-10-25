// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.XPU;

namespace Nncase.Evaluator.TIR.XPU;

public sealed class TransposeEvaluator : ITypeInferencer<Transpose>
{
    public IRType Visit(ITypeInferenceContext context, Transpose target)
    {
        return TupleType.Void;
    }
}
