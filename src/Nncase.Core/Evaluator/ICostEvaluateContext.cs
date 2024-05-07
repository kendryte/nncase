// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Cost evaluate context interface.
/// </summary>
public interface ICostEvaluateContext
{
    /// <summary>
    /// Gets the CompileOptions.
    /// </summary>
    public CompileOptions CompileOptions { get; }

    /// <summary>
    /// Get return type.
    /// </summary>
    /// <typeparam name="T">Return type.</typeparam>
    /// <returns>Casted return type.</returns>
    public T GetReturnType<T>()
     where T : IRType;

    /// <summary>
    /// Get argument type.
    /// </summary>
    /// <typeparam name="T">Argument type.</typeparam>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>Casted argument type.</returns>
    public T GetArgumentType<T>(Op op, ParameterInfo parameter)
     where T : IRType;

    /// <summary>
    /// Get base function argument.
    /// </summary>
    /// <typeparam name="T">Argument type.</typeparam>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>Casted argument type.</returns>
    public T GetArgument<T>(Op op, ParameterInfo parameter)
      where T : BaseFunction;
}
