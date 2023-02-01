// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// if(xxx) then { zzz } else { yyy }.
/// </summary>
public sealed record IfThenElse(Expr Condition, Sequential Then, Sequential Else) : Expr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="IfThenElse"/> class.
    /// ctor.
    /// </summary>
    public IfThenElse(Expr condition, Sequential then)
        : this(condition, then, new Sequential())
    {
    }
}
