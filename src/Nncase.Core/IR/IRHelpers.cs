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
        GC.Collect();
    }

    [Conditional("DEBUG")]
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
            Trace.Assert(user.Users.Any());
        }
    }

    public static void ReplaceAllUsesWith(Expr old, Expr @new)
    {
        old.ReplaceAllUsesWith(@new);
    }

    public static HashSet<DimVar> GetDynamicDimVars()
    {
        return CompileSessionScope.GetCurrentThrowIfNull().CompileOptions.ShapeBucketOptions.VarMap.SelectMany(x => x.Value).OfType<DimVar>().ToHashSet((IEqualityComparer<DimVar>)ReferenceEqualityComparer.Instance);
    }

    public static string GetIdentityName(string name)
    {
        var sb = new StringBuilder("id_");
        foreach (var c in name)
        {
            if (char.IsLetterOrDigit(c) || c == '_')
            {
                sb.Append(c);
            }
            else
            {
                sb.Append('_');
            }
        }

        return sb.ToString();
    }
}
