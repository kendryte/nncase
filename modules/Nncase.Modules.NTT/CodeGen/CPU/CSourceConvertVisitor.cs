// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define MULTI_CORE_CPU

// #define PROFILE_CALL
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

public struct IndentScope : IDisposable
{
    private static readonly AsyncLocal<IndentWriter?> _writer = new AsyncLocal<IndentWriter?>();

    private readonly bool _initialized;

    private readonly IndentWriter? _originalWriter;

    public IndentScope(StringBuilder sb)
    {
        _initialized = true;
        _originalWriter = _writer.Value;
        _writer.Value = new IndentWriter(sb);
    }

    public IndentScope()
    {
        _initialized = true;
        if (_writer.Value is null)
        {
            return;
        }

        _originalWriter = _writer.Value;
        _writer.Value = new(_originalWriter.GetStringBuilder(), _originalWriter.Indent + 2);
    }

    public static IndentWriter Writer => _writer.Value!;

    public void Dispose()
    {
        if (_initialized)
        {
            _writer.Value = _originalWriter;
        }
    }
}

/// <summary>
/// the c symbol define.
/// </summary>
public sealed class CSymbol
{
    public CSymbol(string type, string name)
    {
        Type = type;
        Name = name;
    }

    public static IReadOnlyList<CSymbol> Builtns => new CSymbol[] {
        new CSymbol("nncase_mt_t*", "nncase_mt"),
        new CSymbol("uint8_t*", "data"),
        new CSymbol("const uint8_t*", "rdata"),
    };

    public string Type { get; }

    public string Name { get; }

    public override string ToString() => $"{Type} {Name}";
}

public sealed class IndentWriter : StringWriter
{
    public IndentWriter(StringBuilder sb, int indent = 0)
        : base(sb)
    {
        Indent = indent;
    }

    public int Indent { get; set; }

    public void IndWrite(string? value)
    {
        for (int i = 0; i < Indent; i++)
        {
            Write(' ');
        }

        Write(value);
    }
}

/// <summary>
/// convert single prim function to c source.
/// </summary>
public abstract class CSourceConvertVisitor : ExprFunctor<CSymbol, Unit>
{
    protected readonly Dictionary<BaseExpr, CSymbol> _exprMemo = new(ReferenceEqualityComparer.Instance);

    protected override CSymbol VisitNone(None expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("std::nullptr_t", "nullptr");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimAbs(DimAbs expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("int64_t", $"std::abs({Visit(expr.Operand).Name})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimAt(DimAt expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("int64_t", $"{Visit(expr.Shape).Name}[{Visit(expr.Index).Name}]");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimClamp(DimClamp expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("int64_t", $"std::clamp({Visit(expr.Operand).Name}, {Visit(expr.MinValue).Name}, {Visit(expr.MaxValue).Name})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimCompareAndSelect(DimCompareAndSelect expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("int64_t", $"({Visit(expr.Value).Name} == {Visit(expr.Expected).Name} ? {Visit(expr.TrueValue).Name} : {Visit(expr.FalseValue).Name})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimConst(DimConst expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("int64_t", expr.Value.ToString());
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimFraction(DimFraction expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var numerator = Visit(expr.Numerator).Name;
        var denominator = Visit(expr.Denominator).Name;
        symbol = new("int64_t", expr.DivMode == DimDivideMode.FloorDiv ? $"({numerator} / {denominator})" : $"ntt::ceildiv({numerator}, {denominator})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimMax(DimMax expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var operands = expr.Operands.AsValueEnumerable().Select(x => Visit(x).Name);
        symbol = new("int64_t", $"std::max<int64_t>({{{StringUtility.Join(", ", operands)}}})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimMin(DimMin expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var operands = expr.Operands.AsValueEnumerable().Select(x => Visit(x).Name);
        symbol = new("int64_t", $"std::min<int64_t>({{{StringUtility.Join(", ", operands)}}})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimPositive(DimPositive expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var operand = Visit(expr.Operand).Name;
        var extent = Visit(expr.Extent).Name;
        symbol = new("int64_t", $"({operand} < 0 ? {operand} + {extent} : {operand})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimPower(DimPower expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var dim = Visit(expr.Dim).Name;
        symbol = new("int64_t", $"(int64_t)std::pow({dim}, {expr.Power})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimProduct(DimProduct expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var scale = expr.Scale == 1 ? string.Empty : $"{expr.Scale} * ";
        var operands = expr.Operands.AsValueEnumerable().Select(x => Visit(x).Name);
        symbol = new("int64_t", $"({scale}{StringUtility.Join(" * ", operands)})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimRemainder(DimRemainder expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var numerator = Visit(expr.Numerator).Name;
        var denominator = Visit(expr.Denominator).Name;
        symbol = new("int64_t", $"({numerator} % {denominator})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimSum(DimSum expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var bias = expr.Bias == 0 ? string.Empty : $"{expr.Bias} + ";
        var operands = expr.Operands.AsValueEnumerable().Select(x => Visit(x).Name);
        symbol = new("int64_t", $"({bias}{StringUtility.Join(" + ", operands)})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimVar(DimVar expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("int64_t", expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }
}
