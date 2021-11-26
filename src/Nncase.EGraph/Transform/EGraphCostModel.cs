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

        public Dictionary<EClass, (Cost, ENode)> Context { get => _context; }

        public EGraphCosts(EGraph eGraph, Dictionary<EClass, (Cost, ENode)> context)
        {
            _eGraph = eGraph;
            _context = context;
            foreach (var (enode, eclass) in _eGraph.Nodes)
            {
                _exprMap[enode.Expr] = enode;
            }
        }

        public (Cost, ENode) this[EClass eClass] => _context[eClass.Find()];

        public Cost this[Expr expr] => _context[_eGraph.Nodes[_exprMap[expr]].Find()].Item1;

        public EClass this[ENode eNode] => _eGraph.Nodes[eNode];

    }
}

namespace Nncase.Transform
{


    public sealed partial class EGraph
    {
        public EGraphCosts Costs()
        {
            var visitor = new EGraphCostModelVisitor();
            return new(this, visitor.Visit(this));
        }
    }
}