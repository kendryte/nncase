// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class ConstantOfShapeEvaluator : ITypeInferencer<ConstantOfShape>
{
    public IRType Visit(ITypeInferenceContext context, ConstantOfShape target)
    {
        return TupleType.Void;
    }
}
