// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

public sealed class ExprPinner : IDisposable
{
    private readonly Expr[] _exprs;
    private readonly ExprUser _user = new();
    private bool _disposed;

    public ExprPinner(params Expr[] exprs)
    {
        _exprs = exprs;

        foreach (var expr in _exprs)
        {
            expr.AddUser(_user);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            foreach (var expr in _exprs)
            {
                expr.RemoveUser(_user);
            }

            _disposed = true;
        }
    }
}

internal sealed class ExprUser : Expr
{
    public ExprUser()
        : base(Array.Empty<Expr>())
    {
    }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => throw new NotSupportedException();
}
