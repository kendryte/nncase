using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Transform;
using Nncase.CostModel;

namespace Nncase.CostModel
{
    public record EGraphCosts
    {
        private readonly EGraph _eGraph;
        private readonly Dictionary<EClass, (Cost, ENode)> _context;
        private readonly Dictionary<Expr, ENode> _exprMap = new();
        public EGraphCosts(EGraph eGraph, Dictionary<EClass, (Cost, ENode)> context)
        {
            _eGraph = eGraph;
            _context = context;
            foreach (var (enode, eclass) in _eGraph.Nodes)
            {
                _exprMap[enode.Expr] = enode;
            }
        }
        public Cost this[Expr expr] => _context[_eGraph.Nodes[_exprMap[expr]]].Item1;
    }
}

namespace Nncase.Transform
{

    // public sealed partial class EClass
    // {
    //     public Cost Cost { get; set; }
    // }

    public sealed partial class EGraph
    {
        public EGraphCosts Costs()
        {
            var visitor = new EGraphCostModelVisitor();
            return new(this, visitor.Visit(this));
        }

        public Expr Extract(EClass eClass)
        {
            var visitor = new EGraphCostModelVisitor();
            var result = visitor.Visit(this);
            var converter = new ExprConverter(result);
            return converter.Visit(eClass.Find());
        }

        private sealed class ExprConverter
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
                _ => DefaultVisit(eNode.Expr)
            };

            public Expr DefaultVisit(Expr expr)
            {
                throw new NotImplementedException($"Unhandled visit routine for {expr.GetType()}.");
            }
        }
    }
}