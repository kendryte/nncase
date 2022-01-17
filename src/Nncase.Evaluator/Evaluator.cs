using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics.Tensors;
using System.Reflection;
using Autofac;
using Nncase.Evaluator.Ops;
using Nncase.IR;
using TorchSharp;
using Nncase.IR;
using IContainer = Autofac.IContainer;

namespace Nncase.Evaluator
{
    public interface IEvaluator<T> {}
    
    public static class Evaluator
    {
        public static IContainer Container;

        public static IEnumerable<Type> GetAllEvaluator()
        {
            return Assembly.GetExecutingAssembly()
                .GetTypes()
                .Where(
                    t => t
                        .GetInterfaces()
                        .Any(
                            x => x.IsGenericType && x.GetGenericTypeDefinition() == typeof(IEvaluator<>)));
        }

        internal static void RegisterAll(IEnumerable<Type> evaluators)
        {
            var builder = new ContainerBuilder();
            foreach (var evaluator in evaluators)
            {
                builder.RegisterType(evaluator).AsImplementedInterfaces();
            }
            Container = builder.Build();
        }

        public static void AutoRegister()
        {
            RegisterAll(GetAllEvaluator());
        }
        
        internal static (EvaluatorVisitor, Const) EvalImpl(Expr expr, Dictionary<Var, Const> inputs)
        {
            Container.Init(AutoRegister);
            if (expr.CheckedType is null) expr.InferenceType();
            if (expr.CheckedType is InvalidType)
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
                EvalImpl(func, new()).Item2.ToTorchTensor(),
            Op op => throw new NotImplementedException("Can't Eval a Op!"),
            Var v => throw new NotImplementedException("Can't Eval a Var!"),
            _ => EvalImpl(expr, new()).Item2.ToTorchTensor()
        };

        public static List<Const> Eval(this Expr expr, Dictionary<Var, Const> inputs)
        {
            var visitor = EvalImpl(expr, inputs).Item1;
            var result = new List<Const>();
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
                switch (expr)
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
                        result.Add(visitor.ExpressionMemo[expr]);
                        break;
                }
            }
            return result;
        }
    }
}