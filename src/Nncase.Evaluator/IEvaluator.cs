// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Evaluator interface.
/// </summary>
public interface IEvaluator
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="ctx">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    Const Visit(EvaluatorContext ctx, Op target);
}

/// <summary>
/// Evaluator interface.
/// </summary>
public interface IEvaluator<T> : IEvaluator
    where T : Op
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="ctx">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    Const Visit(EvaluatorContext ctx, T target);

    Const IEvaluator.Visit(EvaluatorContext ctx, Op target)
    {
        return Visit(ctx, (T)target);
    }
}
