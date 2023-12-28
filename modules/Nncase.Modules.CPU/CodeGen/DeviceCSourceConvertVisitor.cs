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
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.TIR;

namespace Nncase.CodeGen.CPU;

internal sealed class DeviceCSourceConvertVisitor : ExprFunctor<CSymbol, Unit>
{
    private readonly Dictionary<Expr, CSymbol> _exprMemo;
    private readonly StringBuilder _deviceBuilder;

    public DeviceCSourceConvertVisitor()
    {
        _exprMemo = new(ReferenceEqualityComparer.Instance);
        _deviceBuilder = new();
    }

    public PrimFunction VisitEntry => (TIR.PrimFunction)VisitRoot!;

    public string GetHeader()
    {
        return _deviceBuilder.ToString();
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPrimFunction(PrimFunction expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.CheckedType is not CallableType { ReturnType: TupleType r } || r != TupleType.Void)
        {
            throw new NotSupportedException("The PrimFunction must return void!");
        }

        var ctype = $"void {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit).Select(s => $"{s.Type} &{s.Name}").ToArray())})";

        using (var scope = new IndentScope(_deviceBuilder))
        {
            // 1. Function signature
            IndentScope.Writer.IndWrite($"{ctype} {{\n");

            // 2. Function body
            using (_ = new IndentScope())
            {
                IndentScope.Writer.IndWrite($"thread_context ctx(bid, tid);\n");
                Visit(expr.Body);
            }

            // 3. Function closing
            IndentScope.Writer.IndWrite("}\n");
        }

        symbol = new(ctype, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitIfThenElse(IfThenElse expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var cond = Visit(expr.Condition);
        IndentScope.Writer.IndWrite($"if ({cond.Name}) {{\n");
        using (_ = new IndentScope())
        {
            Visit(expr.Then);
        }

        IndentScope.Writer.IndWrite("}\n");
        IndentScope.Writer.IndWrite("else {\n");
        using (_ = new IndentScope())
        {
            Visit(expr.Else);
        }

        IndentScope.Writer.IndWrite("}\n");

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitLet(Let expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var @var = Visit(expr.Var);
        var value = Visit(expr.Expression);

#if DEBUG_PRINT
        IndentScope.Writer.IndWrite($"runtime_util->printf(\"let {@var.Name}\\n\");\n");
#endif
        IndentScope.Writer.IndWrite($"{value.Type} {@var.Name} = {value.Name};\n");
        Visit(expr.Body);

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitMemSpan(MemSpan expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var start = Visit(expr.Start);
        _ = Visit(expr.Size);
        string name = expr.Location switch
        {
            MemoryLocation.L2Data => start.Name,
            MemoryLocation.Input or MemoryLocation.Output => start.Name,
            _ => throw new NotSupportedException(expr.Location.ToString()),
        };

        symbol = new(start.Type, name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBuffer(TIR.Buffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var loc = expr.MemSpan.Location is MemoryLocation.Input or MemoryLocation.Output ? MemoryLocation.L1Data.ToC() : expr.MemSpan.Location.ToC();
        var type = $"tensor<{expr.ElemType.ToC()}, {loc}> ";

        symbol = new(type, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitCall(Call expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        string type = expr.CheckedType switch
        {
            TupleType x when x == TupleType.Void => string.Empty,
            TensorType { IsScalar: true } x => x.DType.ToC(),
            _ => throw new NotSupportedException(),
        };

        string str = string.Empty;
        var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        switch (expr.Target)
        {
            case PrimFunction deviceFunc:
                IndentScope.Writer.IndWrite($"{deviceFunc.Name}({string.Join(",", arguments.Select(arg => arg.Name))});\n");
                break;
            case IR.Math.Binary op:
                str = CSourceUtilities.ContertBinary(op, arguments);
                break;
            case IR.Math.Unary op:
                str = CSourceUtilities.ContertUnary(op, arguments);
                break;
            case IR.Math.Compare op:
                str = CSourceUtilities.ContertCompare(op, arguments);
                break;
            case IR.Math.Select op:
                str = CSourceUtilities.ContertSelect(op, arguments);
                break;
            case TIR.CPU.SramPtr op:
                str = $"({type})(sram[bid] + (sram_size_per_thread * tid) + {arguments[0].Name})";
                break;
            case TIR.Load op:
                str = $"{arguments[0].Name}[{arguments[1].Name}]";
                break;
            case TIR.Store op:
#if DEBUG_PRINT
                IndentScope.Writer.IndWrite($"runtime_util->printf(\"{arguments[0].Name}[%d]\\n\", {arguments[1].Name});\n");
#endif
                IndentScope.Writer.IndWrite($"{arguments[0].Name}[{arguments[1].Name}] = {arguments[2].Name};\n");
                break;
            case TIR.CPU.PtrOf op:
                str = op.PtrName + ".data()";
                break;
            case IR.Buffers.Allocate op:
                str = $"({type})runtime_util->malloc({arguments[0].Name})";
                break;
            case IR.Tensors.Cast op:
                str = $"(({op.NewType.ToC()}) {arguments[0].Name})";
                break;
            default:
                throw new NotSupportedException();
        }

        symbol = new(type, str);
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
            str = scalar[0].ToString() switch
            {
                "True" => "1",
                "False" => "0",
                null => string.Empty,
                var x => x,
            };

            type = ptype.ToC();
        }
        else if (expr is TensorConst { Value: Tensor { ElementType: PointerType { ElemType: PrimType }, Shape: { IsScalar: true } } pointer })
        {
            str = pointer.ToScalar<ulong>().ToString();
            type = pointer.ElementType.ToC();
        }
        else
        {
            throw new NotSupportedException();
        }

        symbol = new(type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitSequential(Sequential expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        foreach (var field in expr.Fields)
        {
            Visit(field);
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitFor(For expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        // 1. For Loop signature
        var loopVar = Visit(expr.LoopVar);
        IndentScope.Writer.IndWrite($"for ({loopVar.Type} {loopVar.Name} = {Visit(expr.Domain.Start).Name}; {loopVar.Name} < {Visit(expr.Domain.Stop).Name}; {loopVar.Name}+={Visit(expr.Domain.Step).Name}) {{\n");
#if DEBUG_PRINT
        IndentScope.Writer.IndWrite($"runtime_util->printf(\"{loopVar.Name} = %d\\n\", {loopVar.Name});\n");
#endif

        using (_ = new IndentScope())
        {
            // 2. For Body
            Visit(expr.Body);
        }

        // 3. For closing
        IndentScope.Writer.IndWrite("}\n");

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitVar(Var expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new(
            expr.CheckedType switch
            {
                TensorType t => t.DType.ToC(),
                _ => throw new ArgumentOutOfRangeException(nameof(expr)),
            },
            expr.Name + expr.GlobalVarIndex.ToString());
        _exprMemo.Add(expr, symbol);
        return symbol;
    }
}
