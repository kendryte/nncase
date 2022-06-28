// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CommonServiceLocator;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;

namespace Nncase.Evaluator;

public class DumpManager
{
    public static bool OpenDump { get; private set; } = false;

    public static T RunWithDump<T>(Func<T> f)
    {
        OpenDump = true;
        var result = f();
        OpenDump = false;
        return result;
    }
}
internal sealed class EvaluateVisitor : ExprVisitor<IValue, IRType>
{
    private readonly EvaluateContext _context;
    private readonly IReadOnlyDictionary<Var, IValue> _varsValues;

    private int count = 1;

    private void DumpExpr(TensorValue tensorValue, StreamWriter writer)
    {
        var tensor = tensorValue.AsTensor();
        writer.WriteLine(tensor.Shape.ToString());
        // todo:other type
        var dt = tensor.ElementType;
        if (dt == DataTypes.Float32)
        {
            foreach (var v in tensor.ToArray<float>())
            {
                writer.WriteLine(v);
            }
        }
        else if (dt == DataTypes.Int8 || dt == DataTypes.Int32 || dt == DataTypes.Int64)
        {
            foreach (var v in tensor.ToArray<long>())
            {
                writer.WriteLine(v);
            }
        }
        else
        {
            writer.WriteLine($"Unsupported data type{dt}");
        }
    }
    
    private void DumpCall(Expr expr)
    {
        if (expr is Call)
        {
            // tensor / tensors
            var root = "";
            using (var sr = new StreamWriter(root + count))
            {
                var call = (Call)expr;
                var target = call.Target.GetType().Name.ToLower();
                // todo:dump in last??
                using (var order = new StreamWriter(root + "order", true))
                {
                    order.WriteLine(target);
                }
                sr.WriteLine(target);
                sr.WriteLine(call.CheckedType);
                var result = _context.GetValue(call).AsTensors();
                foreach (var tensor in result)
                {
                    DumpExpr(tensor, sr);
                }
            }
            ++count;
        }
        else
        {
            throw new NotSupportedException("only support Call");
        }
    }
    
    public EvaluateVisitor(IReadOnlyDictionary<Var, IValue> varsValues)
    {
        _context = new EvaluateContext(ExpressionMemo);
        _varsValues = varsValues;
        // if (DumpManager.OpenDump)
        // {
        //     RegisterCallback("DumpResult", DumpCall);
        // }
    }

    /// <inheritdoc/>
    public override IValue VisitLeaf(Call expr)
    {
        _context.CurrentCall = expr;
        return expr.Target switch
        {
            Op op => CompilerServices.EvaluateOp(op, _context),
            Function func => CompilerServices.Evaluate(func.Body, func.Parameters.Zip(expr.Parameters).ToDictionary(kv => kv.First, kv => Visit(kv.Second), (IEqualityComparer<Var>)ReferenceEqualityComparer.Instance)),
            _ => throw new NotImplementedException(expr.Target.ToString())
        };
    }

    public override IValue VisitLeaf(Const expr)
    {
        return Value.FromConst(expr);
    }

    public override IValue VisitLeaf(None expr)
    {
        return Value.None;
    }

    public override IValue VisitLeaf(Marker expr)
    {
        return Visit(expr.Target);
    }

    public override IValue VisitLeaf(Op expr)
    {
        // Value of Op is not needed in evaluate context.
        return null!;
    }

    public override IValue Visit(Function expr)
    {
        // Value of Function is not needed in evaluate context.
        return null!;
    }

    public override IValue VisitLeaf(IR.Tuple expr)
    {
        var fields = expr.Fields.Select(x => Visit(x));
        return new TupleValue(fields.ToArray());
    }

    public override IValue VisitLeaf(Var expr)
    {
        if (!_varsValues.TryGetValue(expr, out var result))
        {
            throw new ArgumentException($"Must Set Input For Var {expr.Name}!");
        }

        if (result is null)
        {
            throw new ArgumentException($"Must Set Input For Var {expr.Name}!");
        }

        if (expr.CheckedType is not AnyType && result.Type != expr.CheckedType)
        {
            throw new ArgumentException(
                $"The Var {expr.Name} Require {expr.CheckedType} But Give {result.Type}");
        }

        if (expr.CheckedType is not AnyType)
        {
            if (result.Type is TensorType resultType)
            {
                if (expr.CheckedShape.IsUnranked)
                {
                    return result;
                }
                var s = expr.CheckedShape.Zip(resultType.Shape).ToArray();
                var matchedShape = s.Aggregate(true, (b, dims) => b && (dims.First.IsUnknown || dims.First == dims.Second));
                if (expr.CheckedDataType != resultType.DType || !matchedShape)
                {
                    throw new ArgumentException(
                        $"The Var {expr.Name} Require {expr.CheckedType} But Give {result.Type}");                    
                }
            }
        }

        return result;
    }
}
