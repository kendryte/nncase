// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes;

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
    /// <param name="basefunc_cost_evaluator">base func cost evaluator.</param>
    /// <returns>Extracted root expression.</returns>
    public static Expr Extract(this IEGraph eGraph, EClass root, Evaluator.IBaseFuncCostEvaluator? basefunc_cost_evaluator)
    {
        // 1. set the all expr checked shape
        foreach (var eclass in eGraph.Classes)
        {
            foreach (var nodes in eclass.Nodes)
            {
                if (eclass.CheckedType.CompareTo(nodes.Expr.CheckedType) > 0)
                {
                    nodes.Expr.CheckedType = eclass.CheckedType;
                }
            }
        }

        // 2. start the cost evaluator
        var costModel = new EGraphCostEvaluator(root.Find(), basefunc_cost_evaluator).Evaluate();
        if (DumpScope.Current.IsEnabled(DumpFlags.EGraphCost))
        {
            using var fs = DumpScope.Current.OpenFile(Path.Combine("Costs", $"V{eGraph.Version}.dot"));
            EGraphPrinter.DumpEgraphAsDot(eGraph, costModel, root.Find(), fs);
        }

        return new EGraphExtractor(costModel).Extract(root.Find());
    }

    /// <summary>
    /// find the minCostEnode in eclass.
    /// <remarks>
    /// the marker first.
    /// </remarks>
    /// </summary>
    internal static ENode MinByWithMarker(this EClass eClass, CostModel.EGraphCostModel costModel)
    {
        return eClass.Nodes.OrderBy(e => e.Expr, ENodeTypeComparer.Instance).MinBy(x => x.Expr is Marker ? Cost.Zero : costModel[x])!;
    }

    /// <summary>
    /// find the minCostEnode in eclass skip marker.
    /// </summary>
    internal static ENode MinByWithOutMarker(this EClass eClass, CostModel.EGraphCostModel costModel)
    {
        return eClass.Nodes.Where(e => e.Expr is not Marker).MinBy(x => costModel[x])!;
    }

    internal sealed class ENodeTypeComparer : IComparer<Expr>
    {
        public static readonly ENodeTypeComparer Instance = new();

        public int Compare(Expr? x, Expr? y) => (x, y) switch
        {
            (null, null) => 0,
            (Expr, null) => 1,
            (null, Expr) => -1,
            (Expr, Expr) => GetPriority(x).CompareTo(GetPriority(y)),
        };

        private int GetPriority(Expr x) => x switch
        {
            Marker => 0,
            Const => 1,
            _ => 2,
        };
    }
}

internal class EGraphExtractor
{
    private readonly EGraphCostModel _costModel;
    private readonly Dictionary<EClass, Expr> _eclassMemo = new();
    private readonly Dictionary<EClass, Expr> _markerEclassMemo = new();
    private StreamWriter? _dumpWriter;

    public EGraphExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public Expr Extract(EClass root)
    {
        _dumpWriter = DumpScope.Current.IsEnabled(DumpFlags.EGraphCost)
            ? new StreamWriter(DumpScope.Current.OpenFile($"{nameof(EGraphExtractor)}_Class_{root.Id}.txt"))
            : null;
        try
        {
            Visit(root);
        }
        finally
        {
            _dumpWriter?.Dispose();
        }

        return _eclassMemo[root];
    }

    private void Visit(EClass eclass)
    {
        var stack = new Stack<(EClass, ENode)>();
        stack.Push((eclass, eclass.MinByWithMarker(_costModel)));
        var markerEclassSet = new HashSet<EClass>();
        while (stack.Any())
        {
            (eclass, var minCostEnode) = stack.Peek();
            if (_eclassMemo.ContainsKey(eclass))
            {
                stack.Pop();
                continue;
            }

            Expr? expr = null;
            switch (minCostEnode.Expr)
            {
                case Var or TensorConst or TupleConst or Op or Fusion or None:
                    expr = minCostEnode.Expr;
                    break;
                case Function or Call or IR.Tuple or Marker or IR.If:
                    var childrenExprs = new List<Expr>();
                    foreach (var child in minCostEnode.Children)
                    {
                        if (!_eclassMemo.TryGetValue(child, out var childExpr))
                        {
                            if (minCostEnode.Expr is Marker && child == eclass)
                            {
                                if (!_markerEclassMemo.TryGetValue(eclass, out var markerInputExpr))
                                {
                                    markerEclassSet.Add(eclass);
                                    stack.Push((eclass, eclass.MinByWithOutMarker(_costModel)));
                                }
                                else
                                {
                                    childrenExprs.Add(markerInputExpr);
                                }
                            }
                            else
                            {
                                stack.Push((child, child.MinByWithMarker(_costModel)));
                            }
                        }
                        else
                        {
                            childrenExprs.Add(childExpr);
                        }
                    }

                    if (childrenExprs.Count != minCostEnode.Children.Count)
                    {
                        break;
                    }

                    expr = minCostEnode.Expr switch
                    {
                        Function function => Visit(minCostEnode, function, new(childrenExprs)),
                        Call call => Visit(minCostEnode, call, new(childrenExprs)),
                        IR.Tuple tuple => Visit(minCostEnode, tuple, new(childrenExprs)),
                        Marker marker => Visit(minCostEnode, marker, new(childrenExprs)),
                        IR.If @if => Visit(minCostEnode, @if, new(childrenExprs)),
                        _ => throw new ArgumentException("Unsupported expression type."),
                    };

                    break;
                default:
                    throw new ArgumentException("Unsupported expression type.");
            }

            if (expr is null)
            {
                continue;
            }

            if (markerEclassSet.Contains(eclass) && minCostEnode.Expr is not Marker)
            {
                _markerEclassMemo.Add(eclass, expr);
            }
            else
            {
                _eclassMemo.Add(eclass, expr);
            }

            stack.Pop();
        }
    }

    private Marker Visit(ENode enode, Marker marker, IRArray<Expr> children)
    {
        var target = children[0];
        var attr = children[1];
        return marker.With(target: target, attribute: attr);
    }

    private Function Visit(ENode enode, Function func, IRArray<Expr> children)
    {
        var body = children[0];
        return func.With(body: body);
    }

    private IR.Tuple Visit(ENode enode, IR.Tuple tuple, IRArray<Expr> children)
    {
        return tuple.With(fields: children.ToArray());
    }

    private IR.If Visit(ENode enode, IR.If @if, IRArray<Expr> children)
    {
        return @if.With(condition: children[0], then: children[1], @else: children[2]);
    }

    private Call Visit(ENode enode, Call call, IRArray<Expr> children)
    {
        var target = children[0];
        var arguments = children.Skip(1);

        // for mix quant debug.
        if (call.EnodeQuantConfigWithCosine != null && _dumpWriter != null)
        {
            _dumpWriter.WriteLine(call + "  " + call.CheckedType);
            for (int i = 0; i < call.EnodeQuantConfigWithCosine.Count; i++)
            {
                for (int j = 0; j < call.EnodeQuantConfigWithCosine[i].Item1.Count; j++)
                {
                    _dumpWriter.Write(call.EnodeQuantConfigWithCosine[i].Item1[j] + "  ");
                }

                _dumpWriter.WriteLine(call.EnodeQuantConfigWithCosine[i].Item3);
            }
        }

        return call.With(target: target, arguments: arguments.ToArray(), call.Metadata);
    }
}
