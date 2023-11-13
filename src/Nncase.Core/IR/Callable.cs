// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// the Callable Expr.
/// </summary>
public abstract class Callable : Expr
{
    /// <summary>
    /// StackVM module kind.
    /// </summary>
    public const string StackVMModuleKind = "stackvm";

    public Callable(string name, string moduleKind, Expr[] operands)
        : base(operands)
    {
        Name = name;
        ModuleKind = moduleKind;
    }

    public string Name { get; }

    public string ModuleKind { get; }
}
