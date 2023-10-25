// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.XPU;

namespace Nncase.Evaluator.TIR.XPU;

public class Conv2DEvaluator : ITypeInferencer<Conv2D>
{
    public IRType Visit(ITypeInferenceContext context, Conv2D target) => TupleType.Void;
}
