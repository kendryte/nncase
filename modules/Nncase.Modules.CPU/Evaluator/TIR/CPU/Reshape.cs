// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class ReshapeEvaluator : ITypeInferencer<Reshape>
{
    public IRType Visit(ITypeInferenceContext context, Reshape target)
    {
        context.CheckArgumentType<TensorType>(target, Reshape.Input);
        context.CheckArgumentType<TensorType>(target, Reshape.Output);
        return TupleType.Void;
    }
}
