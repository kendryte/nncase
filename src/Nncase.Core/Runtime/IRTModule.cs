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
/// Runtime module interface.
/// </summary>
public interface IRTModule
{
    /// <summary>
    /// Gets functions.
    /// </summary>
    IReadOnlyList<IRTFunction> Functions { get; }

    /// <summary>
    /// Initialize module.
    /// </summary>
    /// <returns>Awaitable task.</returns>
    ValueTask InitializeAsync();

    /// <summary>
    /// Uninitialize module.
    /// </summary>
    /// <returns>Awaitable task.</returns>
    ValueTask UninitializeAsync();
}
