using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics.Tensors;
using System.Reflection;
using Autofac;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator;

internal sealed class EvaluateProvider : IEvaluateProvider
{
    //public static torch.Tensor Eval(this Expr expr) => expr switch
    //{
    //    IR.Tuple tuple => throw new NotImplementedException("Can't Eval a Tuple!"),
    //    Function func => func.Body is IR.Tuple ?
    //        throw new NotImplementedException("Can't Eval Function With Retrun Tuple!") :
    //        EvalImpl(func, new()).Item2.ToTorchTensor(),
    //    Op op => throw new NotImplementedException("Can't Eval a Op!"),
    //    Var v => throw new NotImplementedException("Can't Eval a Var!"),
    //    _ => EvalImpl(expr, new()).Item2.ToTorchTensor()
    //};

    //public static List<Const> Eval(this Expr expr, Dictionary<Var, Const> inputs)
    //{
    //    var visitor = EvalImpl(expr, inputs).Item1;
    //    var result = new List<Const>();
    //    if (expr is Function func)
    //    {
    //        switch (func.Body)
    //        {
    //            case (IR.Tuple tuple):
    //                foreach (var item in tuple.Fields)
    //                {
    //                    if (item is IR.Tuple)
    //                        throw new NotImplementedException("Can't Support Return Tuple[Tuple[x,y,...]]!");
    //                    result.Add(visitor.ExpressionMemo[item]);
    //                }

    //                break;
    //            default:
    //                result.Add(visitor.ExpressionMemo[func.Body]);
    //                break;
    //        }
    //    }
    //    else
    //    {
    //        switch (expr)
    //        {
    //            case (IR.Tuple tuple):
    //                foreach (var item in tuple.Fields)
    //                {
    //                    if (item is IR.Tuple)
    //                        throw new NotImplementedException("Can't Support Return Tuple[Tuple[x,y,...]]!");
    //                    result.Add(visitor.ExpressionMemo[item]);
    //                }

    //                break;
    //            default:
    //                result.Add(visitor.ExpressionMemo[expr]);
    //                break;
    //        }
    //    }

    //    return result;
    //}
    private readonly IServiceProvider _serviceProvider;

    public EvaluateProvider(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public Const Evaluate(Expr expr, IReadOnlyDictionary<Var, Const>? varsValues = null)
    {
        if (expr.CheckedType is null)
        {
            expr.InferenceType();
        }

        if (expr.CheckedType is InvalidType)
        {
            throw new InvalidOperationException("Expr in Evaluator need a valid type");
        }

        var evaluatorVisitor = new EvaluateVisitor(varsValues ?? new Dictionary<Var, Const>());
        return evaluatorVisitor.Visit(expr);
    }

    public Const EvaluateOp(Op op, IEvaluateContext context)
    {
        // TODO: Add inferencers cache.
        var evaluatorType = typeof(IEvaluator<>).MakeGenericType(op.GetType());
        var evaluator = (IEvaluator)_serviceProvider.GetRequiredService(evaluatorType);
        return evaluator.Visit(context, op);
    }
}
