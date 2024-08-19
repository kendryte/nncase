// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class ErfEvaluator : ITypeInferencer<Erf>
{
    public IRType Visit(ITypeInferenceContext context, Erf target)
    {
        context.CheckArgumentType<TensorType>(target, Erf.Input);
        context.CheckArgumentType<TensorType>(target, Erf.Output);
        return TupleType.Void;
    }
}
