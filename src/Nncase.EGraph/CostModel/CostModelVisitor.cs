using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Transform;

namespace Nncase.CostModel
{
    // todo:refactor, many copy
    public sealed class EGraphCostModelContext
    {
        private readonly Dictionary<ENode, Cost> _eNodeCosts;
        private readonly Dictionary<EClass, Cost> _eClassCosts;

        public EGraphCostModelContext(Dictionary<ENode, Cost> eNodeCosts, Dictionary<EClass, Cost> eClassCosts)
        {
            _eNodeCosts = eNodeCosts;
            _eClassCosts = eClassCosts;
        }

        public Cost GetArgument(ENode eNode) => _eNodeCosts[eNode];
        
        public Cost GetArgument(EClass eClass) => _eClassCosts[eClass];
    }
    
    public class EGraphVisitor<ENodeResultType, EClassResultType, EGraphResultType>
    {
        private readonly Dictionary<ENode, ENodeResultType> _eNodeMemo;
        private readonly Dictionary<EClass, EClassResultType> _eClassMemo;
        
        public Dictionary<ENode, ENodeResultType> ENodeMemo;
        
        public Dictionary<EClass, EClassResultType> EClassMemo;

        public virtual ENodeResultType VisitLeaf(ENode eNode) => throw new NotImplementedException();
        
        public virtual EClassResultType VisitLeaf(EClass eClass) => throw new NotImplementedException();
        
        public virtual EGraphResultType VisitLeaf(EGraph eGraph) => throw new NotImplementedException();
        
        private ENodeResultType Visit(ENode eNode)
        {
            if (!_eNodeMemo.TryGetValue(eNode, out var result))
            {
                result = VisitLeaf(eNode);
                _eNodeMemo.Add(eNode, result);
            }

            return result;
        }

        private EClassResultType Visit(EClass eClass)
        {
            if (!_eClassMemo.TryGetValue(eClass, out var result))
            {
                eClass.Select(Visit);
                result = VisitLeaf(eClass);
                _eClassMemo.Add(eClass, result);
            }

            return result;
        }
        
        private EGraphResultType Visit(EGraph eGraph)
        {
            eGraph.Select(Visit);
            return VisitLeaf(eGraph);
        }
    }

    public sealed class EGraphCostModelVisitor : EGraphVisitor<Cost, Cost, Cost>
    {
        private EGraphCostModelContext _context;
        
        public EGraphCostModelVisitor()
        {
            _context = new EGraphCostModelContext(ENodeMemo, EClassMemo);
        }
        
        public override Cost VisitLeaf(ENode eNode)
        {
            // todo: maybe error
            var exprVisitor = new ExprCostModelVisitor();
            return exprVisitor.Visit(eNode.Expr);
        }

        public override Cost VisitLeaf(EClass eClass)
        {
            return eClass.Select(x => _context.GetArgument(x)).Min();
        }

        public override Cost VisitLeaf(EGraph eGraph)
        {
            throw new NotImplementedException();
        }
    }

    public class ExprCostModelContext
    {
        public Call? CurrentCall { get; set; }

        public Expr GetArgument(Op op, ParameterInfo parameter)
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

        public Const GetArgumentConst(Op op, ParameterInfo parameter)
        {
            if(GetArgument(op, parameter) is Const constValue)
            {
                return constValue;
            }
            else
            {
                throw new InvalidOperationException($"Op:{op} Parameter:{parameter} is not const");
            }
        }

        public TensorType GetTensorType(Expr expr)
        {
            var resultType = expr.CheckedType?? throw new InvalidOperationException($"Expr {expr} don't have CheckedType.");
            return resultType is TensorType resultTensorType ? 
                resultTensorType :
                throw new InvalidOperationException($"Expr {expr} is not a TensorType.");
        }

        public TensorType GetArgumentType(Op op, ParameterInfo parameter) =>
            GetTensorType(GetArgument(op, parameter));

        public TensorType CurrentCallResultTensorType()
        {
            return GetTensorType(CurrentCall ?? throw new ArgumentException($"Current Call {CurrentCall} is null"));
        }
        
        private Call GetCurrentCall() => CurrentCall ?? throw new InvalidOperationException("Current call is not set.");
    }
    
    public sealed partial class ExprCostModelVisitor : ExprVisitor<Cost, IRType>
    {
        private readonly ExprCostModelContext _context;
        public ExprCostModelVisitor()
        {
            _context = new ExprCostModelContext();
        }

        public override Cost VisitLeaf(Call expr)
        {
            _context.CurrentCall = expr;
            return expr.Target switch
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

        public override Cost VisitLeaf(Const expr)
        {
            return new Cost();
        }
        
        public override Cost VisitLeaf(Op expr)
        {
            return new Cost();
        }

        public override Cost VisitLeaf(Function expr)
        {
            return new Cost();
        }
        
        public override Cost VisitLeaf(IR.Tuple expr)
        {
            return new Cost();
        }
        
        public override Cost VisitLeaf(Var expr)
        {
            return new Cost();
        }
    }
}