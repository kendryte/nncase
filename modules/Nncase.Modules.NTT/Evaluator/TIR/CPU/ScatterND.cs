﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class ScatterNDEvaluator : ITypeInferencer<ScatterND>
{
    public IRType Visit(ITypeInferenceContext context, ScatterND target)
    {
        return TupleType.Void;
    }
}
