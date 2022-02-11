// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Transform;

namespace Nncase.CostModel
{
    public class EGraphVisitor<ENodeResultType, EClassResultType, EGraphResultType>
    {
        private readonly Dictionary<ENode, ENodeResultType> _eNodeMemo = new();
        private readonly Dictionary<EClass, EClassResultType> _eClassMemo = new();

        public Dictionary<ENode, ENodeResultType> ENodeMemo { get => _eNodeMemo; }

        public Dictionary<EClass, EClassResultType> EClassMemo { get => _eClassMemo; }

        public virtual ENodeResultType VisitLeaf(ENode eNode) => throw new NotImplementedException();

        public virtual EClassResultType VisitLeaf(EClass eClass) => throw new NotImplementedException();

        public virtual EGraphResultType VisitLeaf(EGraph eGraph) => throw new NotImplementedException();

        public virtual ENodeResultType Visit(ENode eNode) => throw new NotImplementedException();

        public virtual EClassResultType Visit(EClass eClass) => throw new NotImplementedException();

        public virtual EGraphResultType Visit(EGraph eGraph)
        {
            return VisitLeaf(eGraph);
        }
    }

    public sealed record ENodeCost(Cost Partial, Cost Total)
    {
    }

    public sealed record EClassCost(bool Locked, Cost Total)
    {
    }

    public sealed class EGraphCostModelContext : ExprCostModelContext
    {
        public readonly Dictionary<ENode, ENodeCost> _eNodeCosts;
        public readonly Dictionary<EClass, EClassCost> _eClassCosts;
        public IReadOnlyDictionary<EClass, List<ENode>> _eClasses;
        public IReadOnlyDictionary<ENode, EClass> _eHashCon;
        public readonly Dictionary<EClass, IRType> _eClassType;
        public readonly Dictionary<EClass, (Cost, ENode)> CostEnv;

        /// <summary>
        /// find the expr's Enode for ExprCostVisitor.
        /// </summary>
        public readonly Dictionary<Expr, ENode> _exprMaps;

        public EGraphCostModelContext(Dictionary<ENode, ENodeCost> eNodeCosts, Dictionary<EClass, EClassCost> eClassCosts)
        {
            _eNodeCosts = eNodeCosts;
            _eClassCosts = eClassCosts;
            _exprMaps = new();
            _eClassType = new();
            CostEnv = new();
            _eClasses = new Dictionary<EClass, List<ENode>>();
            _eHashCon = new Dictionary<ENode, EClass>();
        }

        public override Cost GetCostFromMemo(Expr expr, int i)
        {
            return _eClassCosts[_exprMaps[expr].Children[i]].Total;
        }

        private EClass GetParentEClass(Expr expr) => _eHashCon[_exprMaps[expr]].Find();

        public override TensorType CurrentCallResultTensorType() =>
          CheckTensorType(_eClassType[GetParentEClass(CurrentCall)]);

        public override Const GetArgumentConst(Op op, ParameterInfo parameter)
        {
            var curEclass = _exprMaps[CurrentCall].Children[parameter.Index + 1];
            foreach (var node in _eClasses[curEclass])
            {
                if (node.Expr is Const c)
                {
                    return c;
                }
            }

            throw new InvalidOperationException("This EClass Does Not Have Const Enode!");
        }

        public override TensorType GetArgumentType(Op op, ParameterInfo parameter)
        {
            if (op.GetType() == parameter.OwnerType)
            {
                return _eClassType[_exprMaps[CurrentCall].Children[parameter.Index + 1]] switch
                {
                    TensorType ttype => ttype,
                    _ => throw new InvalidOperationException($"Expr is not a TensorType."),
                };
            }

            throw new InvalidOperationException($"The {op.GetType().Name} has now parameter {parameter.Name}!");
        }

        public override TensorType GetTensorType(Expr expr) => CheckTensorType(_eClassType[GetParentEClass(expr)]);

        private TensorType CheckTensorType(IRType type) => type switch
        {
            TensorType ttype => ttype,
            _ => throw new InvalidOperationException($"Expr is not a TensorType."),
        };
    }

    public sealed class EGraphCostModelVisitor : EGraphVisitor<ENodeCost, EClassCost, Dictionary<EClass, (Cost, ENode)>>
    {
        private EGraphCostModelContext _context;
        public readonly ExprCostModelVisitor ExprVisitor;

        public EGraphCostModelVisitor()
        {
            _context = new EGraphCostModelContext(ENodeMemo, EClassMemo);
            ExprVisitor = new ExprCostModelVisitor(_context);
        }

        private bool Changed = true;

        public string Indent = "";

        public override EClassCost Visit(EClass eClass)
        {
            var old_cost = EClassMemo[eClass];
            if (old_cost.Locked)
                return old_cost;
            _context._eClasses[eClass].ForEach(node => Visit(node));
            var new_cost = VisitLeaf(eClass);
            if (old_cost.Total != new_cost.Total)
            {
                Changed = true;
                EClassMemo[eClass] = old_cost with { Total = new_cost.Total };
            }

            return new_cost;
        }

        public override EClassCost VisitLeaf(EClass eClass)
        {
            var ENodes = _context._eClasses[eClass];
            var min = Cost.Inf;
            ENode minNode = ENodes[0];
            foreach (var (eNode, i) in Enumerable.Range(0, ENodes.Count).Select(i => (ENodes[i], i)))
            {
                if (min > _context._eNodeCosts[eNode].Total)
                {
                    min = _context._eNodeCosts[eNode].Total;
                    minNode = eNode;
                }
            }

            /* if is leaf eclass, wo can lock it */
            var eClassCost = new EClassCost((ENodes.Count == 1 && minNode.Children.Count == 0),
                                            ENodeMemo[minNode].Total);
            _context.CostEnv[eClass] = (min, minNode);
            return eClassCost;
        }

        public override ENodeCost Visit(ENode eNode)
        {
            if (!ENodeMemo.TryGetValue(eNode, out var result))
            {
                result = VisitLeaf(eNode);
            }

            var total = result.Partial;
            foreach (var child in eNode.Children)
            {
                total += EClassMemo[child].Total;
            }

            ENodeMemo[eNode] = result with { Total = total };
            return result;
        }

        public override ENodeCost VisitLeaf(ENode eNode)
        {
            var partial = ExprVisitor.VisitLeaf(eNode.Expr);
            return new ENodeCost(partial, Cost.Inf);
        }

        public override Dictionary<EClass, (Cost, ENode)> Visit(EGraph eGraph)
        {
            _context._eClasses = eGraph.EClasses();
            _context._eHashCon = eGraph.HashCons;

            // make expr map and eclass shape map
            foreach (var (eClass, eNodes) in _context._eClasses)
            {
                // the same Eclass must have same IRType
                var types = new List<IRType>();
                foreach (var eNode in eNodes)
                {
                    _context._exprMaps.Add(eNode.Expr, eNode);
                    types.Add(eNode.Expr.CheckedType!);
                }

                var vaild_types = types.Distinct().ToArray();
                if (vaild_types.Length != 1)
                {
                    throw new InvalidProgramException("The Same EClass Must Have Same IRType!");
                }

                _context._eClassType[eClass] = vaild_types[0];
                EClassMemo[eClass] = new EClassCost(false, Cost.Inf);
            }

            return VisitLeaf(eGraph);
        }

        public override Dictionary<EClass, (Cost, ENode)> VisitLeaf(EGraph eGraph)
        {
            while (Changed)
            {
                Changed = false;
                foreach (var (eClass, eNodes) in _context._eClasses)
                {
                    Visit(eClass);
                }
            }

            return _context.CostEnv;
        }
    }
}