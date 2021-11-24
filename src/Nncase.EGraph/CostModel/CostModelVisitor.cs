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
    public class ExprCostModelContext
    {
        public virtual Call? CurrentCall { get; set; }

        public Dictionary<Expr, Cost> ExpressionMemo { get; set; } = new();

        public virtual Call GetCurrentCall() => CurrentCall ?? throw new InvalidOperationException("Current call is not set.");

        public virtual Expr GetArgument(Op op, ParameterInfo parameter)
        {
            if (op.GetType() == parameter.OwnerType)
            {
                return GetCurrentCall().Parameters[parameter.Index];
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
            }
        }

        public virtual Const GetArgumentConst(Op op, ParameterInfo parameter)
        {
            if (GetArgument(op, parameter) is Const constValue)
            {
                return constValue;
            }
            else
            {
                throw new InvalidOperationException($"Op:{op} Parameter:{parameter} is not const");
            }
        }

        public virtual TensorType GetTensorType(Expr expr)
        {
            var resultType = expr.CheckedType ?? throw new InvalidOperationException($"Expr {expr} don't have CheckedType.");
            return resultType is TensorType resultTensorType ?
                resultTensorType :
                throw new InvalidOperationException($"Expr {expr} is not a TensorType.");
        }

        public virtual TensorType GetArgumentType(Op op, ParameterInfo parameter) =>
            GetTensorType(GetArgument(op, parameter));

        public virtual TensorType CurrentCallResultTensorType()
        {
            return GetTensorType(CurrentCall ?? throw new ArgumentException($"Current Call {CurrentCall} is null"));
        }

        public virtual Cost GetCostFromMemo(Expr expr, int i) => ExpressionMemo[expr];
    }

    public sealed class EGraphCostModelContext : ExprCostModelContext
    {
        public readonly Dictionary<ENode, Cost> _eNodeCosts;
        public readonly Dictionary<EClass, Cost> _eClassCosts;

        public Dictionary<EClass, List<ENode>> _eClasses;

        /// <summary>
        /// find the expr's Enode for ExprCostVisitor.
        /// </summary>
        public readonly Dictionary<Expr, ENode> _exprMaps;

        public Cost TotalCost = Cost.Inf();

        public EGraphCostModelContext(Dictionary<ENode, Cost> eNodeCosts, Dictionary<EClass, Cost> eClassCosts)
        {
            _eNodeCosts = eNodeCosts;
            _eClassCosts = eClassCosts;
            _exprMaps = new();
        }

        public override Cost GetCostFromMemo(Expr expr, int i)
        {
            return _eClassCosts[_exprMaps[expr].Children[i]];
        }
    }

    public class EGraphVisitor<ENodeResultType, EClassResultType, EGraphResultType>
    {
        private readonly Dictionary<ENode, ENodeResultType> _eNodeMemo = new();
        private readonly Dictionary<EClass, EClassResultType> _eClassMemo = new();

        public Dictionary<ENode, ENodeResultType> ENodeMemo { get => _eNodeMemo; }

        public Dictionary<EClass, EClassResultType> EClassMemo { get => _eClassMemo; }

        public virtual ENodeResultType VisitLeaf(ENode eNode) => throw new NotImplementedException();

        public virtual EClassResultType VisitLeaf(EClass eClass) => throw new NotImplementedException();

        public virtual EGraphResultType VisitLeaf(EGraph eGraph) => throw new NotImplementedException();

        protected virtual ENodeResultType Visit(ENode eNode) => throw new NotImplementedException();

        protected virtual EClassResultType Visit(EClass eClass) => throw new NotImplementedException();

        protected virtual EGraphResultType Visit(EGraph eGraph)
        {
            return VisitLeaf(eGraph);
        }
    }

    public sealed class EGraphCostModelVisitor : EGraphVisitor<Cost, Cost, Dictionary<EClass, (Cost, ENode)>>
    {
        private EGraphCostModelContext _context;
        public readonly ExprCostModelVisitor ExprVisitor;

        public EGraphCostModelVisitor()
        {
            _context = new EGraphCostModelContext(ENodeMemo, EClassMemo);
            ExprVisitor = new ExprCostModelVisitor(_context);
        }
        private bool Changed = true;

        protected override Cost Visit(EClass eClass)
        {
            return _context._eClassCosts[eClass];
        }

        protected override Cost Visit(ENode eNode)
        {
            if (eNode.Children.Count == 0)
            {
                if (!_context._eNodeCosts.TryGetValue(eNode, out var leaf_result))
                {   // when chileren=0, expr will be op/const/var
                    // we can get cost directly
                    leaf_result = ExprVisitor.VisitLeaf(eNode.Expr);
                    _context._eNodeCosts.Add(eNode, leaf_result);
                }
                return leaf_result;
            }
            eNode.Children.Select(Visit);
            // visit call/func/tuple
            var result = VisitLeaf(eNode);
            _context._eNodeCosts[eNode] = result;
            return result;
        }

        public override Cost VisitLeaf(ENode eNode)
        {
            _context.CurrentCall = (eNode.Expr is Call call ? call : null);
            return ExprVisitor.VisitLeaf(eNode.Expr);
        }

        public override Dictionary<EClass, (Cost, ENode)> VisitLeaf(EGraph eGraph)
        {
            _context._eClasses = eGraph.EClasses();
            var eClassSeq = eGraph.TopSort(_context._eClasses);

            // make expr map
            foreach (var (_, eNodes) in _context._eClasses)
            {
                foreach (var eNode in eNodes)
                {
                    _context._exprMaps.Add(eNode.Expr, eNode);
                }
            }

            while (Changed)
            {
                foreach (var eClass in eClassSeq)
                {
                  foreach (var eNode in _context._eClasses[eClass])
                  {
                      
                  }
                }
            }

            // while (Changed)
            // {
            //     Changed = false;
            //     foreach (var (eClass, eNodes) in _context._eClasses)
            //     {
            //         var new_cost = eNodes.Select(Visit).Min()!;
            //         if (EClassMemo[eClass] != new_cost)
            //         {
            //             Changed = true;
            //         }
            //         EClassMemo[eClass] = new_cost;
            //     }
            // }
            var CostEnv = new Dictionary<EClass, (Cost, ENode)>();
            return CostEnv;
        }
    }

    public sealed partial class ExprCostModelVisitor : ExprVisitor<Cost, IRType>
    {
        private readonly ExprCostModelContext _context;

        public ExprCostModelVisitor()
        {
            _context = new ExprCostModelContext();
            _context.ExpressionMemo = ExpressionMemo;
        }

        public ExprCostModelVisitor(ExprCostModelContext context)
        {
            _context = context;
            _context.ExpressionMemo = ExpressionMemo;
        }

        public override Cost VisitLeaf(Call expr)
        {
            _context.CurrentCall = expr;
            return _context.GetCostFromMemo(expr.Target, 0) +
               expr.Target switch
               {
                   Binary bn => VisitBinary(bn),
                   // Concat con => VisitConcat(con),
                   Conv2D conv => VisitConv2D(conv),
                   // Slice sl => VisitSlice(sl),
                   // Transpose tr => VisitTranspose(tr),
                   Unary un => VisitUnary(un),
                   ShapeOp => throw new InvalidDataException("ShapeOp should be eliminate before CostModelVisitor"),
                   _ => throw new NotImplementedException()
               };
        }

        public override Cost VisitLeaf(Const expr) =>
         new Cost(0, expr.CheckedShape.Size *
             DataTypes.GetLength(expr.ValueType.DType));


        public override Cost VisitLeaf(Op expr) => new Cost(1);

        public override Cost VisitLeaf(Function expr) =>
          _context.GetCostFromMemo(expr.Body, 0) +
           expr.Parameters.Select((p, i) => _context.GetCostFromMemo(p, 1 + i))
            .Aggregate((l, r) => l + r) + new Cost(1);


        public override Cost VisitLeaf(IR.Tuple expr) =>
          expr.Fields.Select((p, i) => _context.GetCostFromMemo(p, i)).
            Aggregate((l, r) => l + r);

        public override Cost VisitLeaf(Var expr) => new Cost();

    }
}