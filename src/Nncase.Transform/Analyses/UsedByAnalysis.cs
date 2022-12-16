using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Nncase.IR;

namespace Nncase.Transform.Analyses;

internal sealed class UsedByAnalysisVisitor : ExprVisitor<bool, IRType>
{
    private BaseFunction? _entry = null;

    public readonly Dictionary<Expr, HashSet<Expr>> UseByMap;

    public UsedByAnalysisVisitor()
    {
        UseByMap = new(ReferenceEqualityComparer.Instance);
    }

    public static void AddUsedBy(Dictionary<Expr, HashSet<Expr>> map, Expr child, Expr parent)
    {
        if (child is Call { Target: Fusion { Name: "fusion_2_True_fusion_1_True_fusion_0_True" } })
            System.Console.WriteLine("fuck!");
        if (!map.TryGetValue(child, out var chain))
        {
            chain = new(ReferenceEqualityComparer.Instance);
            map.Add(child, chain);
        }
        chain.Add(parent);
    }

    public static bool ClearUsedBy(Dictionary<Expr, HashSet<Expr>> map, Expr child, Expr parent)
    {
        var users = map[child];
        var ret = users.Remove(parent);
        if (users.Count == 0)
            map.Remove(child);
        return ret;
    }

    public override bool Visit(BaseFunction baseFunction)
    {
        if (_entry is null)
        {
            _entry = baseFunction;
        }
        else
            return false;
        return base.Visit(baseFunction);
    }

    public override bool DefaultVisitLeaf(Expr expr) => false;

    public override bool VisitLeaf(Call expr)
    {
        AddUsedBy(UseByMap, expr.Target, expr);
        foreach (var param in expr.Parameters)
            AddUsedBy(UseByMap, param, expr);

        // create the chain for current call
        if (!UseByMap.TryGetValue(expr, out var chain))
        {
            chain = new(ReferenceEqualityComparer.Instance);
            UseByMap.Add(expr, chain);
        }
        return false;
    }

    public static IUsedByResult Analysis(Expr entry)
    {
        var vistor = new UsedByAnalysisVisitor();
        vistor.Visit(entry);
        return new SimpleDuChain(vistor.UseByMap);
    }
}

internal sealed class SimpleDuChain : IUsedByResult
{
    public Dictionary<Expr, HashSet<Expr>> UseByMap;

    public IReadOnlyDictionary<Expr, HashSet<Expr>> MeMo => UseByMap;

    public SimpleDuChain(Dictionary<Expr, HashSet<Expr>> du_chain)
    {
        UseByMap = du_chain;
    }

    public HashSet<Expr> Get(Expr child) => UseByMap[child];

    public void Clear(Expr child, Expr parent)
    {
        UsedByAnalysisVisitor.ClearUsedBy(UseByMap, child, parent);
    }

    public void Add(Expr child, Expr parent)
    {
        UsedByAnalysisVisitor.AddUsedBy(UseByMap, child, parent);
    }

    public void Transfer(Expr old_expr, Expr new_expr)
    {
        var old_usedby = UseByMap[old_expr];
        if (!UseByMap.TryGetValue(new_expr, out var new_usedby))
        {
            new_usedby = new(old_usedby, ReferenceEqualityComparer.Instance);
            UseByMap.Add(new_expr, new_usedby);
        }
        else
            throw new ArgumentOutOfRangeException("The new_call is not new created call");
    }
}