﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
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
    BaseCall CurrentCall { get; }

    /// <summary>
    /// Get argument value.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    IValue GetArgumentValue(Op op, ParameterInfo parameter);

    IValue Evaluate(BaseExpr expr);

    public Option<IValue> GetOptionalArgumentValue(Op op, ParameterInfo parameter)
    {
        var v = GetArgumentValue(op, parameter);
        return v is NoneValue ? Option.None : Option.Some(v);
    }

    /// <summary>
    /// Get argument value as Tensor{T}.
    /// </summary>
    /// <typeparam name="T">Scalar type.</typeparam>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    public Tensor<T> GetArgumentValueAsTensor<T>(Op op, ParameterInfo parameter)
        where T : struct, IEquatable<T>
    {
        return GetArgumentValue(op, parameter).AsTensor().Cast<T>();
    }

    /// <summary>
    /// Get argument value as Tensor.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    public Tensor GetArgumentValueAsTensor(Op op, ParameterInfo parameter)
    {
        return GetArgumentValue(op, parameter).AsTensor();
    }

    /// <summary>
    /// Get argmument value as Tensors.
    /// </summary>
    public Tensor[] GetArgumentValueAsTensors(Op op, ParameterInfo parameter)
    {
        return GetArgumentValue(op, parameter).AsTensors();
    }

    /// <summary>
    /// Get argument value as scalar.
    /// </summary>
    /// <typeparam name="T">Scalar type.</typeparam>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    public T GetArgumentValueAsScalar<T>(Op op, ParameterInfo parameter)
        where T : struct, IEquatable<T>
    {
        return GetArgumentValue(op, parameter).AsTensor().ToScalar<T>();
    }

    public T GetOptionArgumentValueAsScalar<T>(Op op, ParameterInfo parameter, T dft)
        where T : struct, IEquatable<T>
    {
        return GetOptionalArgumentValue(op, parameter).Match(
            x => x.AsTensor().ToScalar<T>(),
            () => dft);
    }

    /// <summary>
    /// Get argument value as array.
    /// </summary>
    /// <typeparam name="T">Scalar type.</typeparam>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument value.</returns>
    public T[] GetArgumentValueAsArray<T>(Op op, ParameterInfo parameter)
        where T : struct, IEquatable<T>
    {
        return GetArgumentValue(op, parameter).AsTensor().ToArray<T>();
    }
}
