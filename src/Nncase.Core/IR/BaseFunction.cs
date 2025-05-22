﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Base function.
/// </summary>
public abstract class BaseFunction : Callable
{
    public BaseFunction(string name, string moduleKind, BaseExpr[] operands)
        : base(name, moduleKind, operands)
    {
        SchedResult = new();
    }

    /// <summary>
    /// Gets or sets sched result.
    /// </summary>
    public Schedule.SchedFunctionResult SchedResult { get; set; }

    /// <summary>
    /// Gets parameter types.
    /// </summary>
    public abstract IEnumerable<IRType> ParameterTypes { get; }

    public bool IsEntry { get; set; }

    public abstract BaseFunction With(string? name = null, string? moduleKind = null);
}
