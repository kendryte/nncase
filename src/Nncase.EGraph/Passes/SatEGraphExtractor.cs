// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Google.OrTools.Sat;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.Passes;

internal class SatEGraphExtractor : IEGraphExtractor
{
    private readonly EGraphCostModel _costModel;
    private readonly Dictionary<EClass, Expr> _eclassMemo = new();
    private readonly Dictionary<EClass, Expr> _markerEclassMemo = new();

    public SatEGraphExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public Expr Extract(EClass root, IEGraph eGraph)
    {
        var cpmodel = new CpModel();

        // 0. create bool var for all enode.
        var vars = new Dictionary<ENode, BoolVar>();
        foreach (var item in eGraph.Nodes.Select((e, i) => (e, i)))
        {
            vars.Add(item.e, cpmodel.NewBoolVar(item.i.ToString()));
        }

        // 1. must pick one root enode.
        cpmodel.AddExactlyOne(root.Nodes.Select(n => vars[n]).ToArray());

        // 2. when pick node, must pick one child node.
        foreach (var n in eGraph.Nodes)
        {
            var ns = new[] { vars[n].Not() };
            foreach (var child in n.Children)
            {
                cpmodel.AddBoolOr(ns.Concat(child.Nodes.Select(cn => vars[cn])));
            }
        }

        // 3. if eclass contains marker have two or more enodes, do not pick marker.
        foreach (var eclass in eGraph.Classes)
        {
            if (eclass.Nodes.Count <= 1)
            {
                continue;
            }

            foreach (var mknode in eclass.Nodes.Where(e => e.Expr is Marker).ToArray())
            {
                cpmodel.AddAssumption(vars[mknode].Not());
            }
        }

        // 3. add pick weights for all enode.
        cpmodel.Minimize(LinearExpr.WeightedSum(eGraph.Nodes.Select(n => vars[n]), eGraph.Nodes.Select(n => checked((long)_costModel[n].Score))));

        if (cpmodel.Validate().Any())
        {
            throw new InvalidDataException("the sat model invalid: " + cpmodel.Validate());
        }

        var solver = new CpSolver();
        var status = solver.Solve(cpmodel);

        if (status is not (CpSolverStatus.Optimal or CpSolverStatus.Optimal))
        {
            return new EGraphExtractor(_costModel).Extract(root, eGraph);
        }

        return new SatExprBuildVisitor(eGraph.Nodes.ToDictionary(e => e, e => solver.BooleanValue(vars[e]))).Visit(root);
    }
}

internal sealed class SatExprBuildVisitor
{
    private readonly IReadOnlyDictionary<ENode, bool> _pick;
    private readonly Dictionary<EClass, Expr> _memo;

    public SatExprBuildVisitor(IReadOnlyDictionary<ENode, bool> pick)
    {
        _pick = pick;
        _memo = new();
    }

    public Expr Visit(EClass root)
    {
        Expr? expr;
        if (_memo.TryGetValue(root, out expr))
        {
            return expr;
        }

        var enodes = root.Nodes.Where(n => _pick[n]).ToArray();
        if (enodes.Length != 1)
        {
            throw new InvalidProgramException("the one eclass only can pick one enode!");
        }

        var enode = enodes[0];
        var children = enode.Children.Select(e => Visit(e)).ToArray();

        switch (enode.Expr)
        {
            case Var or TensorConst or TupleConst or Op or Fusion or None:
                expr = enode.Expr;
                break;
            case Function func:
                expr = func.With(body: children[0], parameters: children[1..].OfType<Var>().ToArray());
                break;
            case Call call:
                expr = call.With(target: children[0], arguments: children[1..], call.Metadata);
                break;
            case IR.Tuple tp:
                expr = tp.With(fields: children);
                break;
            case Marker mk:
                expr = mk.With(target: children[0], attribute: children[1]);
                break;
            case IR.If @if:
                expr = @if.With(condition: children[0], then: children[1], @else: children[2]);
                break;
            default:
                throw new NotSupportedException(enode.Expr.GetType().Name);
        }

        _memo.Add(root, expr);

        return expr;
    }
}
