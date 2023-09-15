// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.XPU;

namespace Nncase.Evaluator.XPU;

public sealed class LayerNormEvaluator : ITypeInferencer<LayerNorm>
{
    public IRType Visit(ITypeInferenceContext context, LayerNorm target) => TupleType.Void;
}
