﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Runtime;

/// <summary>
/// Runtime model interface.
/// </summary>
public interface IRTModel
{
    /// <summary>
    /// Gets modules.
    /// </summary>
    IReadOnlyList<IRTModule> Modules { get; }

    /// <summary>
    /// Gets entry function.
    /// </summary>
    IRTFunction Entry { get; }

    /// <summary>
    /// Initialize model.
    /// </summary>
    /// <returns>Awaitable task.</returns>
    ValueTask InitializeAsync();

    /// <summary>
    /// Uninitialize model.
    /// </summary>
    /// <returns>Awaitable task.</returns>
    ValueTask UninitializeAsync();
}
