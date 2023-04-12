// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// Evaluator for <see cref="Nop"/>.
/// </summary>
public class NopEvaluator : ITypeInferencer<Nop>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Nop target)
    {
        return TupleType.Void;
    }
}
