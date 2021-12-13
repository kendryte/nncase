using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using Nncase.Evaluator.Ops;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator
{
    public static class Evaluator
    {
        internal static (EvaluatorVisitor, torch.Tensor) EvalImpl(Expr expr, Dictionary<Var, torch.Tensor> inputs)
        {
            if (expr.CheckedType is null or InvalidType)
                throw new InvalidOperationException("Expr in Evaluator need a valid type");

            var evaluatorVisitor = new EvaluatorVisitor(inputs);
            var result = evaluatorVisitor.Visit(expr);
            return (evaluatorVisitor, result);
        }

        public static torch.Tensor Eval(this Expr expr) => expr switch
        {
            IR.Tuple tuple => throw new NotImplementedException("Can't Eval a Tuple!"),
            Function func => func.Body is IR.Tuple ?
                throw new NotImplementedException("Can't Eval Function With Retrun Tuple!") :
                EvalImpl(func, new()).Item2,
            Op op => throw new NotImplementedException("Can't Eval a Op!"),
            Var v => throw new NotImplementedException("Can't Eval a Var!"),
            _ => EvalImpl(expr, new()).Item2
        };

        public static List<Const> Eval(this Expr expr, Dictionary<Var, Const> args)
        {
            var inputs = args.ToDictionary(pair => pair.Key, pair => pair.Value.ToTorchTensor());
            var visitor = EvalImpl(expr, inputs).Item1;
            var result = new List<torch.Tensor>();
            if (expr is Function func)
            {
                switch (func.Body)
                {
                    case (IR.Tuple tuple):
                        foreach (var item in tuple.Fields)
                        {
                            if (item is IR.Tuple)
                                throw new NotImplementedException("Can't Support Return Tuple[Tuple[x,y,...]]!");
                            result.Add(visitor.ExpressionMemo[item]);
                        }
                        break;
                    default:
                        result.Add(visitor.ExpressionMemo[func.Body]);
                        break;
                }
            }
            else
            {
                throw new NotImplementedException($"Can't Support Eval {expr.GetType().Name} with inputs!");
            }
            return result.Select(t => t.ToConst()).ToList();
        }
    }
}