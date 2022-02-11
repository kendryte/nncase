// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using Nncase.Transform.Rule;
using Nncase.IR;
using Nncase.Pattern;
using System.Linq;
using Nncase.CostModel;

namespace Nncase.Transform
{
    public static class EGraphExtractor
    {
        public static Expr Extract(this EGraph eGraph, EClass entry, RunPassOptions options)
        {
            var visitor = new EGraphCostModelVisitor();
            var costs = visitor.Visit(eGraph);
            var converter = new ExprConverter(costs);
            if (options.DumpLevel > 1)
            {
                EGraphPrinter.DumpEgraphAsDot(eGraph, new EGraphCosts(eGraph, costs), entry.Find(), Path.Combine(options.PassDumpDir, "Costs", $"V{eGraph.Version}"));
            }

            return converter.Visit(entry.Find());
        }
    }

    internal sealed class ExprConverter
    {
        private Dictionary<EClass, (Cost, ENode)> _context;
        private readonly Dictionary<ENode, Expr> _enodeMemo = new();

        private readonly Dictionary<EClass, Expr> _eclassMemo = new
        ();

        public ExprConverter(Dictionary<EClass, (Cost, ENode)> context)
        {
            _context = context;
        }

        public Expr Visit(EClass eClass)
        {
            if (!_eclassMemo.TryGetValue(eClass, out var result))
            {
                Visit(_context[eClass].Item2);
                result = VisitLeaf(eClass);
                _eclassMemo.Add(eClass, result);
            }

            return result;
        }

        public Expr VisitLeaf(EClass eClass)
        {
            return _enodeMemo[_context[eClass].Item2];
        }

        public Expr Visit(ENode eNode)
        {
            if (!_enodeMemo.TryGetValue(eNode, out var result))
            {
                foreach (var eClass in eNode.Children)
                {
                    Visit(eClass);
                }

                result = VisitLeaf(eNode);
                _enodeMemo.Add(eNode, result);
            }

            return result;
        }

        public Expr VisitLeaf(ENode eNode) => eNode.Expr switch
        {
            Var var => var,
            Const con => con,
            Function func => new Function(_eclassMemo[eNode.Children[0]], eNode.Children.Skip(1).Select(p => _eclassMemo[p]).ToArray()),
            Call call => new Call(_eclassMemo[eNode.Children[0]], eNode.Children.Skip(1).Select(p => _eclassMemo[p]).ToArray()),
            IR.Tuple tuple => new IR.Tuple(eNode.Children.Select(p => _eclassMemo[p]).ToArray()),
            Op op => op,
            _ => DefaultVisit(eNode.Expr),
        };

        public Expr DefaultVisit(Expr expr)
        {
            throw new NotImplementedException($"Unhandled visit routine for {expr.GetType()}.");
        }
    }
}