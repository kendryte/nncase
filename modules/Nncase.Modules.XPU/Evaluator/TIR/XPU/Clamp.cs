// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.XPU;

namespace Nncase.Evaluator.TIR.XPU;

public sealed class ClampEvaluator : ITypeInferencer<Clamp>
{
    public IRType Visit(ITypeInferenceContext context, Clamp target) => TupleType.Void;
}
