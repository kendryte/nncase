// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.CPU;
using Nncase.Schedule;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class SUMMAEvaluator : ITypeInferencer<SUMMA>
{
    public IRType Visit(ITypeInferenceContext context, SUMMA target) => TupleType.Void;
}
