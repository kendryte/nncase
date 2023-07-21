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
/// Metric evaluator interface.
/// </summary>
public interface IMetricEvaluator
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    Metric Visit(IMetricEvaluateContext context, Op target);
}

/// <summary>
/// BaseFunction Evaluator Metric interface.
/// </summary>
public interface IBaseFuncMetricEvaluator
{
    /// <summary>
    /// Evaluate the Base Function Metric.
    /// </summary>
    /// <param name="target">Target Fusion/Primfunc/PrimfuncWrapper.</param>
    /// <returns>The base function Metrics.</returns>
    Metric VisitLeaf(BaseFunction target);
}

/// <summary>
/// Metric evaluator interface.
/// </summary>
public interface IMetricEvaluator<T> : IMetricEvaluator
    where T : Op
{
    /// <summary>
    /// Evaluate op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    Metric Visit(IMetricEvaluateContext context, T target);

    Metric IMetricEvaluator.Visit(IMetricEvaluateContext context, Op target)
    {
        return Visit(context, (T)target);
    }
}
