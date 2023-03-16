// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Runtime;

/// <summary>
/// Runtime function interface.
/// </summary>
public interface IRTFunction
{
    /// <summary>
    /// Gets parameter types.
    /// </summary>
    IReadOnlyList<IRType> ParameterTypes { get; }

    /// <summary>
    /// Gets return type.
    /// </summary>
    IRType ReturnType { get; }

    /// <summary>
    /// Initialize function.
    /// </summary>
    /// <returns>Awaitable task.</returns>
    ValueTask InitializeAsync();

    /// <summary>
    /// Uninitialize function.
    /// </summary>
    /// <returns>Awaitable task.</returns>
    ValueTask UninitializeAsync();

    /// <summary>
    /// Invoke function.
    /// </summary>
    /// <param name="parameters">Arguments.</param>
    /// <param name="ret">Return value.</param>
    /// <returns>Awaitable task.</returns>
    ValueTask InvokeAsync(IReadOnlyList<IValue> parameters, IValue ret);
}
