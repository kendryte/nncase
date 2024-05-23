﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class SoftmaxEvaluator : ITypeInferencer<Softmax>
{
    public IRType Visit(ITypeInferenceContext context, Softmax target) => TupleType.Void;
}
