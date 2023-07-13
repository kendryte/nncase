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
/// Metric evaluate provider interface.
/// </summary>
public interface IMetricEvaluateProvider
{
    /// <summary>
    /// Evaluate Metric of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Evaluate result.</returns>
    Dictionary<Expr, Metric> EvaluateMetric(Expr expr);

    /// <summary>
    /// Evaluate Metric of operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <returns>Evaluate result.</returns>
    Metric EvaluateOpMetric(Op op, IMetricEvaluateContext context);
}
