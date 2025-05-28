// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class ShapeOfEvaluator : ITypeInferencer<ShapeOf>
{
    public IRType Visit(ITypeInferenceContext context, ShapeOf target)
    {
        return TupleType.Void;
    }
}
