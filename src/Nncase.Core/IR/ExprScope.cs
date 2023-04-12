// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

internal sealed class ExprScope : IDisposable
{
    private static readonly AsyncLocal<ExprScope?> _exprScope = new AsyncLocal<ExprScope?>();

    private readonly ExprScope? _originalExprScope;
    private readonly List<Expr> _exprs = new();

    public ExprScope()
    {
        _originalExprScope = _exprScope.Value;
        _exprScope.Value = this;
    }

    public static ExprScope? Current => _exprScope.Value;

    public IReadOnlyList<Expr> Exprs => _exprs;

    public void Add(Expr expr) => _exprs.Add(expr);

    public void Dispose()
    {
        _exprScope.Value = _originalExprScope;
    }
}
