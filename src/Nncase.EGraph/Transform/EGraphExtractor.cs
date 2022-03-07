// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

internal class EGraphExtractor
{
    private readonly EGraphCostModel _costModel;
    private readonly IReadOnlyDictionary<EClass, List<ENode>> _eclasses;
    private readonly Dictionary<EClass, Expr> _eclassMemo = new();

    public EGraphExtractor(EGraphCostModel costModel, IReadOnlyDictionary<EClass, List<ENode>> eclasses)
    {
        _costModel = costModel;
        _eclasses = eclasses;
    }

    public Expr Extract(EClass root)
    {
        return Visit(root);
    }

    private Expr Visit(EClass eclass)
    {
        if (!_eclassMemo.TryGetValue(eclass, out var expr))
        {
            var enodes = _eclasses[eclass];
            var minCostEnode = enodes.MinBy(x => _costModel[x]);
            expr = minCostEnode!.Expr switch
            {
                Var var => VisitLeaf(minCostEnode, var),
                TensorConst con => VisitLeaf(minCostEnode, con),
                TupleConst con => VisitLeaf(minCostEnode, con),
                Function func => VisitLeaf(minCostEnode, func),
                Call call => Visit(minCostEnode, call),
                IR.Tuple tuple => Visit(minCostEnode, tuple),
                Op op => VisitLeaf(minCostEnode, op),
                _ => throw new ArgumentException("Unsupported expression type."),
            };
            _eclassMemo.Add(eclass, expr);
        }

        return expr;
    }

    private Expr VisitLeaf(ENode enode, Expr expr)
    {
        return expr;
    }

    private IR.Tuple Visit(ENode enode, IR.Tuple tuple)
    {
        var fields = enode.Children.Select(Visit);
        return new(fields);
    }

    private Call Visit(ENode enode, Call call)
    {
        var target = Visit(enode.Children[0]);
        var parameters = enode.Children.Skip(1).Select(Visit);
        return new(target, parameters.ToArray());
    }
}

/// <summary>
/// EGraph extract extensions.
/// </summary>
public static class EGraphExtractExtensions
{
    /// <summary>
    /// Extract egraph.
    /// </summary>
    /// <param name="eGraph">eGraph.</param>
    /// <param name="root">Root eclass.</param>
    /// <param name="options">Options.</param>
    /// <returns>Extracted root expression.</returns>
    public static Expr Extract(this EGraph eGraph, EClass root, RunPassOptions options)
    {
        var eclasses = eGraph.EClasses();
        var costModel = new EGraphCostEvaluator(eclasses, root).Evaluate();
        if (options.DumpLevel > 1)
        {
            // TODO: dump graph
            // EGraphPrinter.DumpEgraphAsDot(eGraph, new EGraphCosts(eGraph, costs), entry.Find(), Path.Combine(options.PassDumpDir, "Costs", $"V{eGraph.Version}"));
        }

        return new EGraphExtractor(costModel, eclasses).Extract(root);
    }
}
