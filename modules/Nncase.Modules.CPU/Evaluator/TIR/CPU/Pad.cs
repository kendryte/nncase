// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class PadEvaluator : ITypeInferencer<Pad>
{
    public IRType Visit(ITypeInferenceContext context, Pad target)
    {
        context.CheckArgumentType<TensorType>(target, Pad.Input);
        context.CheckArgumentType<TensorType>(target, Pad.Output);
        return TupleType.Void;
    }
}
