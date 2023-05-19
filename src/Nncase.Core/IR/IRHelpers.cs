// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

public static class IRHelpers
{
    public static IRType? GetRawCheckedType(Expr expr) => expr.RawCheckedType;

    public static void SetRawCheckedType(Expr expr, IRType? value) => expr.RawCheckedType = value;

    public static void DCE(BaseFunction function)
    {
        using var exprPin = new ExprPinner(function);
        var exprs = ExprCollector.Collect(function);
        var users = new HashSet<Expr>(ReferenceEqualityComparer.Instance);

        void AddUsers(Expr expr)
        {
            if (expr is not ExprUser
                && expr.IsAlive
                && users.Add(expr))
            {
                foreach (var user in expr.Users)
                {
                    AddUsers(user);
                }
            }
        }

        foreach (var expr in exprs)
        {
            AddUsers(expr);
        }

        foreach (var user in users)
        {
            user.DisposeIfNoUsers();
        }

        DCESanity(function);
    }

    // [Conditional("DEBUG")]
    public static void DCESanity(Expr root)
    {
        using var exprPin = new ExprPinner(root);
        var exprs = ExprCollector.Collect(root);
        var users = new HashSet<Expr>(ReferenceEqualityComparer.Instance);

        void AddUsers(Expr expr)
        {
            if (expr is not ExprUser
                && expr.IsAlive
                && users.Add(expr))
            {
                foreach (var user in expr.Users)
                {
                    AddUsers(user);
                }
            }
        }

        foreach (var expr in exprs)
        {
            AddUsers(expr);
        }

        foreach (var user in users)
        {
            Trace.Assert(user.Users.Count > 0);
        }
    }
}
