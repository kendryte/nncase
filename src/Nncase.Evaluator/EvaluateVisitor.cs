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

    public static bool Append = false;
    
    public static int Count = 1;

    public static string Dir;

    public static void RunWithDump(string dir, Action f)
    {
        RunWithDump<int>(dir, () =>
        {
            f();
            return -1;
        });
    }
    
    public static T RunWithDump<T>(string dir, Func<T> f)
    {
        Dir = dir;
        Count = 1;
        OpenDump = true;
        Append = false;
        var result = f();
        OpenDump = false;
        return result;
    }
}
internal sealed class EvaluateVisitor : ExprVisitor<IValue, IRType>
{
    private readonly EvaluateContext _context;
    private readonly IReadOnlyDictionary<Var, IValue> _varsValues;
    
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

    private static string GetEvaluatorDumpDir()
    {
        var root = Path.Join(CompilerServices.CompileOptions.DumpDir, DumpManager.Dir);
        if (!Directory.Exists(root))
        {
            Directory.CreateDirectory(root);
        }

        return root;
    }

    private void DumpCallInfo(Expr expr)
    {
        if (expr is Call call)
        {
            var root = GetEvaluatorDumpDir();
            var target = call.Target.GetType().Name.ToLower();
            var paramsInfo = ((Op) call.Target).Parameters.ToArray();
            for (int i = 0; i < call.Parameters.Count; i++)
            {
                using (var sr = new StreamWriter(Path.Join(root, DumpManager.Count.ToString() + target + $"_param_{i}_{paramsInfo[i].Name}")))
                {
                    var param = call.Parameters[i];
                    if (param is not IR.Tuple)
                    {
                        var p = _context.GetValue(param).AsTensor();
                        DumpExpr(p, sr);
                    }
                    else
                    {
                        // todo: not impl
                    }
                }
            }
        }
    }
    
    private void DumpCall(Expr expr)
    {
        if (expr is Call call)
        {
            // tensor / tensors
            var root = GetEvaluatorDumpDir();

            var target = call.Target.GetType().Name.ToLower();
            using (var sr = new StreamWriter(Path.Join(root, DumpManager.Count.ToString() + target)))
            {
                sr.WriteLine(target);
                sr.WriteLine(call.CheckedType);
                var result = _context.GetValue(call).AsTensors();
                foreach (var tensor in result)
                {
                    DumpExpr(tensor, sr);
                }
            }
            using (var order = new StreamWriter(Path.Join(root, "order"), DumpManager.Append))
            {
                order.WriteLine(target);
            }
            DumpManager.Append = true;
            ++DumpManager.Count;
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
        if (DumpManager.OpenDump)
        {
            RegisterBeforeCallback("DumpResult", DumpCallInfo);
            RegisterAfterCallback("DumpResult", DumpCall);
        }
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

        if (expr.CheckedType is not AnyType)
        {
            if (result.Type is TensorType resultType)
            {
                if (expr.CheckedShape.IsUnranked)
                {
                    return result;
                }

                if (expr.CheckedDataType != resultType.DType)
                {
                    throw new ArgumentException($"DataType mismatch. The Var {expr.Name} Require {expr.CheckedDataType} But Give {resultType.DType}");
                }
                
                var s = expr.CheckedShape.Zip(resultType.Shape).ToArray();
                var matchedShape = s.Aggregate(true, (b, dims) => b && (dims.First.IsUnknown || dims.Second.IsUnknown || dims.First == dims.Second));
                if(!matchedShape)
                {
                    throw new ArgumentException(
                        $"Shape mismatch. The Var {expr.Name} Require {expr.CheckedShape} But Give {resultType.Shape}");                    
                }
            }
        }

        return result;
    }
}
