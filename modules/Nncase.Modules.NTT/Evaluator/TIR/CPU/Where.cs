// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class WhereEvaluator : ITypeInferencer<Where>
{
    public IRType Visit(ITypeInferenceContext context, Where target) => TupleType.Void;
}
