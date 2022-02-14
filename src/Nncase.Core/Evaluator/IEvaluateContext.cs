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
/// Evaluate context interface.
/// </summary>
public interface IEvaluateContext
{
    /// <summary>
    /// Gets current call expression.
    /// </summary>
    Call CurrentCall { get; }

    /// <summary>
    /// Get argument expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument expression.</returns>
    Expr GetArgumentExpr(Op op, ParameterInfo parameter);

    /// <summary>
    /// Get expression value.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>The value.</returns>
    IValue GetValue(Expr expr);

    /// <summary>
    /// Get argument value.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    public IValue GetArgumentValue(Op op, ParameterInfo parameter)
    {
        var expr = GetArgumentExpr(op, parameter);
        if (expr is Const constValue)
        {
            return Value.FromConst(constValue);
        }
        else
        {
            // maybe a valid type but not const
            return GetValue(expr);
        }
    }

    /// <summary>
    /// Get argument value as scalar.
    /// </summary>
    /// <typeparam name="T">Scalar type.</typeparam>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    public Tensor<T> GetArgumentValueAsTensor<T>(Op op, ParameterInfo parameter)
        where T : unmanaged, IEquatable<T>
    {
        return GetArgumentValue(op, parameter).AsTensor().Cast<T>();
    }

    /// <summary>
    /// Get argument value as scalar.
    /// </summary>
    /// <typeparam name="T">Scalar type.</typeparam>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    public T GetArgumentValueAsScalar<T>(Op op, ParameterInfo parameter)
        where T : unmanaged, IEquatable<T>
    {
        return GetArgumentValue(op, parameter).AsTensor().ToScalar<T>();
    }

    /// <summary>
    /// Get argument value as array.
    /// </summary>
    /// <typeparam name="T">Scalar type.</typeparam>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    public T[] GetArgumentValueAsArray<T>(Op op, ParameterInfo parameter)
        where T : unmanaged, IEquatable<T>
    {
        return GetArgumentValue(op, parameter).AsTensor().ToArray<T>();
    }
}
