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
/// Shape evaluate provider interface.
/// </summary>
public interface IShapeEvaluateProvider
{
    /// <summary>
    /// Evaluate Shape of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="cache">VarMap.</param>
    /// <returns>Evaluate result.</returns>
    Expr EvaluateShapeExpr(Expr expr, ShapeExprCache cache);

    /// <summary>
    /// Evaluate Shape of operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <returns>Evaluate result.</returns>
    Expr EvaluateOpShapeExpr(Op op, IShapeEvaluateContext context);
}
