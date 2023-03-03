// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Nncase.IR;

namespace Nncase.Transform.Analyses;

internal sealed class UsedByAnalysisVisitor : ExprVisitor<bool, IRType>
{
    private readonly Dictionary<Expr, HashSet<Expr>> _useByMap;

    private BaseFunction? _entry;

    public UsedByAnalysisVisitor()
    {
        _useByMap = new(ReferenceEqualityComparer.Instance);
    }

    public static void AddUsedBy(Dictionary<Expr, HashSet<Expr>> map, Expr child, Expr parent)
    {
        if (!map.TryGetValue(child, out var chain))
        {
            chain = new(ReferenceEqualityComparer.Instance);
            map.Add(child, chain);
        }

        chain.Add(parent);
    }

    public static bool ClearUsedBy(Dictionary<Expr, HashSet<Expr>> map, Expr child, Expr parent)
    {
        bool ret = false;
        if (map.TryGetValue(child, out var users))
        {
            ret = users.Remove(parent);
            if (users.Count == 0)
            {
                map.Remove(child);
            }
        }

        return ret;
    }

    public static IUsedByResult Analysis(Expr entry)
    {
        var vistor = new UsedByAnalysisVisitor();
        vistor.Visit(entry);
        return new SimpleDuChain(vistor._useByMap);
    }

    public override bool Visit(BaseFunction baseFunction)
    {
        if (_entry is null)
        {
            _entry = baseFunction;
        }
        else
        {
            return false;
        }

        return base.Visit(baseFunction);
    }

    public override bool DefaultVisitLeaf(Expr expr) => false;

    public override bool VisitLeaf(IR.Tuple expr)
    {
        foreach (var param in expr.Fields)
        {
            AddUsedBy(_useByMap, param, expr);
        }

        // create the chain for current call
        if (!_useByMap.TryGetValue(expr, out _))
        {
            HashSet<Expr>? chain = new(ReferenceEqualityComparer.Instance);
            _useByMap.Add(expr, chain);
        }

        return false;
    }

    public override bool VisitLeaf(Call expr)
    {
        AddUsedBy(_useByMap, expr.Target, expr);
        foreach (var param in expr.Parameters)
        {
            AddUsedBy(_useByMap, param, expr);
        }

        // create the chain for current call
        if (!_useByMap.TryGetValue(expr, out _))
        {
            HashSet<Expr>? chain = new(ReferenceEqualityComparer.Instance);
            _useByMap.Add(expr, chain);
        }

        return false;
    }
}

internal sealed class SimpleDuChain : IUsedByResult
{
    private readonly Dictionary<Expr, HashSet<Expr>> _useByMap;

    public SimpleDuChain(Dictionary<Expr, HashSet<Expr>> du_chain)
    {
        _useByMap = du_chain;
    }

    public IReadOnlyDictionary<Expr, HashSet<Expr>> MeMo => _useByMap;

    public HashSet<Expr> Get(Expr child) => _useByMap[child];

    public void Clear(Expr child, Expr parent)
    {
        UsedByAnalysisVisitor.ClearUsedBy(_useByMap, child, parent);
    }

    public void Add(Expr child, Expr parent)
    {
        UsedByAnalysisVisitor.AddUsedBy(_useByMap, child, parent);
    }

    public void Transfer(Expr old_expr, Expr new_expr)
    {
        var old_usedby = _useByMap[old_expr];
        if (!_useByMap.TryGetValue(new_expr, out _))
        {
            HashSet<Expr>? new_usedby = new(old_usedby, ReferenceEqualityComparer.Instance);
            _useByMap.Add(new_expr, new_usedby);
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(new_expr), "The new_call is not new created call");
        }
    }
}
