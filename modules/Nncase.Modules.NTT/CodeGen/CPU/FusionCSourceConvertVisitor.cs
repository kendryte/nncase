// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define MULTI_CORE_XPU

// #define DEBUG_PRINT
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

public class LambdaCSourceConvertVisitor : CSourceConvertVisitor
{
    protected readonly StringBuilder _fusionBuilder;
    protected int _ssaValueId;

    public LambdaCSourceConvertVisitor()
    {
        _fusionBuilder = new();
        _ssaValueId = 0;
    }

    public new Fusion VisitEntry => (Fusion)VisitRoot!;

    public string GetHeader()
    {
        return _fusionBuilder.ToString();
    }

    protected override CSymbol VisitFusion(Fusion expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.CheckedType is not CallableType { ReturnType: AnyType r })
        {
            throw new NotSupportedException("The LambdaFusion must return auto type!");
        }

        var ctype = expr.Name;

        using (var scope = new IndentScope(_fusionBuilder))
        {
            // 1. Function signature
            IndentScope.Writer.IndWrite($"template<{string.Join(", ", Enumerable.Range(0, expr.Parameters.Length).Select(x => $"class T{x}"))}> struct {expr.Name} {{\n");
            using (_ = new IndentScope())
            {
                IndentScope.Writer.IndWrite($"auto operator()({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit).Select((s, i) => $"const T{i} &{s.Name}").ToArray())}) const noexcept {{\n");

                // 2. Function body
                using (_ = new IndentScope())
                {
                    var ret = Visit(expr.Body);
                    IndentScope.Writer.IndWrite($"return {ret.Name};\n");
                }

                // 3. Function closing
                IndentScope.Writer.IndWrite("}\n");
            }

            IndentScope.Writer.IndWrite("};\n");
        }

        symbol = new(ctype, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitCall(Call expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        string type = "auto";
        string str = string.Empty;
        var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        var name = $"v{_ssaValueId++}";
        switch (expr.Target)
        {
            case IR.Math.Binary binary:
                str = $"{binary.BinaryOp.ToNTT()}({arguments[0].Name}, {arguments[1].Name})";
                break;

            default:
                throw new NotSupportedException($"The call target {expr.Target.GetType()} is not supported in C source code generation.");
        }

        IndentScope.Writer.IndWrite($"{type} {name} = {str};\n");
        symbol = new(type, name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitConst(Const expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        string type;
        string str;
        if (expr is TensorConst { Value: Tensor { ElementType: PrimType ptype, Shape: { IsScalar: true } } scalar })
        {
            str = scalar[Array.Empty<long>()].ToString() switch
            {
                "True" => "1",
                "False" => "0",
                null => string.Empty,
                var x => x,
            };

            type = ptype.ToC();
        }
        else if (expr is TensorConst { Value: Tensor { ElementType: PointerType { ElemType: DataType }, Shape: { IsScalar: true } } pointer })
        {
            str = pointer.ToScalar<ulong>().ToString();
            type = pointer.ElementType.ToC();
        }
        else
        {
            throw new NotSupportedException();
        }

        symbol = new(type, $"{type}({str})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitTupleConst(TupleConst tp)
    {
        if (_exprMemo.TryGetValue(tp, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = $"{string.Join(",", tp.Value.Select(x => Visit(Const.FromValue(x)).Name))}";
        symbol = new(type, str);
        _exprMemo.Add(tp, symbol);
        return symbol;
    }

    protected override CSymbol VisitTuple(IR.Tuple tp)
    {
        if (_exprMemo.TryGetValue(tp, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = $"{string.Join(",", tp.Fields.AsValueEnumerable().Select(x => Visit(x).Name).ToArray())}";
        symbol = new(type, str);
        _exprMemo.Add(tp, symbol);
        return symbol;
    }

    protected override CSymbol VisitVar(Var expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var name = IRHelpers.GetIdentityName(expr.Name);
        var index = VisitEntry.Parameters.IndexOf(expr);
        if (index != -1)
        {
            symbol = new CSymbol($"T{index}", name);
        }
        else
        {
            symbol = new(
                expr.CheckedType switch
                {
                    TensorType t => t.DType.ToC(),
                    AnyType => "auto",
                    _ => throw new ArgumentOutOfRangeException(nameof(expr)),
                },
                expr.Name + "_" + expr.GlobalVarIndex.ToString());
        }

        _exprMemo.Add(expr, symbol);
        return symbol;
    }
}
