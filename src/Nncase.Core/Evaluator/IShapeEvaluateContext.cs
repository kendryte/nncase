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
///  Shape evaluator context interface.
/// </summary>
public interface IShapeEvaluateContext
{
    /// <summary>
    /// Get argument expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument expression.</returns>
    Expr GetArgument(Op op, ParameterInfo parameter);

    /// <summary>
    /// Get arguments expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="paramsInfo">ParamsInfo.</param>
    /// <returns>The arguments expression.</returns>
    Expr[] GetArguments(Op op, params ParameterInfo[] paramsInfo);

    /// <summary>
    /// Get argument shape expr.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument shape expr.</returns>
    Expr GetArgumentShape(Op op, ParameterInfo parameter);


    /// <summary>
    /// Get argument rank expr.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument shape expr.</returns>
    Expr GetArgumentRank(Op op, ParameterInfo parameter);
}
