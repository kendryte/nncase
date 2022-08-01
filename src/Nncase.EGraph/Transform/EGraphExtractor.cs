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
    private readonly Dictionary<EClass, Expr> _eclassMemo = new();

    public EGraphExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public Expr Extract(EClass root)
    {
        return Visit(root);
    }

    private Expr Visit(EClass eclass)
    {
        if (!_eclassMemo.TryGetValue(eclass, out var expr))
        {
            var minCostEnode = eclass.Nodes.MinBy(x => _costModel[x]);
            expr = minCostEnode!.Expr switch
            {
                Var var => VisitLeaf(minCostEnode, var),
                TensorConst con => VisitLeaf(minCostEnode, con),
                TupleConst con => VisitLeaf(minCostEnode, con),
                Function func => Visit(minCostEnode, func),
                Call call => Visit(minCostEnode, call),
                IR.Tuple tuple => Visit(minCostEnode, tuple),
                Op op => VisitLeaf(minCostEnode, op),
                Marker marker => Visit(minCostEnode, marker),
                _ => throw new ArgumentException("Unsupported expression type."),
            };
            _eclassMemo.Add(eclass, expr);
        }
        if (expr.enodequantconfigwithcosine != null)
        {
            System.Console.WriteLine(expr + "  " + expr.CheckedType);
            for (int i = 0; i < expr.enodequantconfigwithcosine.Count; i++)
            {
                for (int j = 0; j < expr.enodequantconfigwithcosine[i].Item1.Count; j++)
                {
                    System.Console.Write(expr.enodequantconfigwithcosine[i].Item1[j] + "  ");
                }
                System.Console.WriteLine(expr.enodequantconfigwithcosine[i].Item2);
            }
        }
        return expr;
    }

    private Expr VisitLeaf(ENode enode, Expr expr)
    {
        return expr;
    }

    private Marker Visit(ENode enode, Marker marker)
    {
        var target = Visit(enode.Children[0]);
        var attr = Visit(enode.Children[1]);
        return marker with { Target = target, Attribute = attr };
    }

    private Function Visit(ENode enode, Function func)
    {
        var body = Visit(enode.Children[0]);
        return func with { Body = body };
    }

    private IR.Tuple Visit(ENode enode, IR.Tuple tuple)
    {
        var fields = enode.Children.Select(Visit);
        return tuple with { Fields = new(fields) };
    }

    private Call Visit(ENode enode, Call call)
    {
        var target = Visit(enode.Children[0]);
        var parameters = enode.Children.Skip(1).Select(Visit);
        return call with { Target = target, Parameters = new(parameters) };
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
        var costModel = new EGraphCostEvaluator(root).Evaluate();
        if (options.DumpLevel > 3)
        {
            // TODO: dump graph
            // EGraphPrinter.DumpEgraphAsDot(eGraph, new EGraphCosts(eGraph, costs), entry.Find(), Path.Combine(options.PassDumpDir, "Costs", $"V{eGraph.Version}"));
        }

        return new EGraphExtractor(costModel).Extract(root);
    }
}
