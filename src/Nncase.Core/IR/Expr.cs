// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance.Helpers;
using Nncase.Diagnostics;

namespace Nncase.IR;

/// <summary>
/// Expression.
/// </summary>
public abstract partial class Expr : BaseExpr
{
    internal Expr(IEnumerable<BaseExpr> operands)
        : base(operands)
    {
    }

    internal Expr(BaseExpr[] operands)
        : base(operands)
    {
    }
}
