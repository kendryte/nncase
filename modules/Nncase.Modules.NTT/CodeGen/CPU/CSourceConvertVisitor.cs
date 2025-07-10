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
using Nncase.IR.Shapes;
using Nncase.Runtime;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

public enum CShapeKind
{
    Shape,
    Strides,
    I64Dims,
}

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

    public PrimFunction VisitEntry => (TIR.PrimFunction)VisitRoot!;

    protected void WriteDimVars()
    {
        var varMap = CreateDimVarMap();
        foreach (var (dimVar, (tensorVar, dimIndex)) in varMap)
        {
            if (!VisitEntry.Parameters.Contains(dimVar))
            {
                var dimVarName = Visit(dimVar).Name;
                var tensorVarName = Visit(tensorVar).Name;
                IndentScope.Writer.IndWrite($"auto {dimVarName} = {tensorVarName}.shape()[{dimIndex}_dim];\n");
                if (dimVar.Metadata.Range is ValueRange<double> range)
                {
                    IndentScope.Writer.IndWrite($"if (!({range.Min} <= {dimVarName} && {dimVarName} <= {range.Max})) {{\n");
                    IndentScope.Writer.IndWrite($"  printf(\"{dimVarName} is out of range [{range.Min}, {range.Max}]\");\n");
                    IndentScope.Writer.IndWrite($"  return;\n");
                    IndentScope.Writer.IndWrite($"}}\n");
                }
            }
        }
    }

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

        symbol = new("dim_t", $"std::abs({Visit(expr.Operand).Name})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimAt(DimAt expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("dim_t", $"{Visit(expr.Shape).Name}[{Visit(expr.Index).Name}]");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimClamp(DimClamp expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("dim_t", $"ntt::clamp({Visit(expr.Operand).Name}, {Visit(expr.MinValue).Name}, {Visit(expr.MaxValue).Name})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimCompareAndSelect(DimCompareAndSelect expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("dim_t", $"({Visit(expr.Value).Name} == {Visit(expr.Expected).Name} ? {Visit(expr.TrueValue).Name} : {Visit(expr.FalseValue).Name})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimConst(DimConst expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new($"fixed_dim<{expr.Value}>", $"{expr.Value}_dim");
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
        symbol = new("dim_t", expr.DivMode == DimDivideMode.FloorDiv ? $"({numerator} / {denominator})" : $"ntt::ceil_div({numerator}, {denominator})");
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
        symbol = new("dim_t", $"ntt::max({StringUtility.Join(", ", operands)})");
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
        symbol = new("dim_t", $"ntt::min({StringUtility.Join(", ", operands)})");
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
        symbol = new("dim_t", $"ntt::positive_index({operand}, {extent})");
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
        symbol = new("dim_t", $"(dim_t)std::pow({dim}, {expr.Power})");
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
        symbol = new("dim_t", $"({scale}{StringUtility.Join(" * ", operands)})");
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
        symbol = new("dim_t", $"({numerator} % {denominator})");
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
        symbol = new("dim_t", $"({bias}{StringUtility.Join(" + ", operands)})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitDimVar(DimVar expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new("dim_t", expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitRankedShape(RankedShape expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var operands = expr.Operands.AsValueEnumerable().Select(Visit);
        var types = StringUtility.Join(", ", operands.Select(x => x.Type));
        var values = StringUtility.Join(", ", operands.Select(x => x.Name));
        symbol = new($"shape_t<{types}>", $"make_shape({values})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitShapeOf(ShapeOf expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var value = Visit(expr.Value).Name;
        symbol = new("auto", $"{value}.shape()");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitPadding(Padding expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var before = Visit(expr.Before).Name;
        var after = Visit(expr.After).Name;
        symbol = new("auto", $"ntt::make_padding({before}, {after})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitPaddings(Paddings expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var operands = expr.Operands.AsValueEnumerable().Select(x => Visit(x).Name);
        symbol = new("auto", $"ntt::make_paddings({StringUtility.Join(", ", operands)})");
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected CSymbol VisitDimOrShape(BaseExpr expr, CShapeKind shapeKind = CShapeKind.I64Dims)
    {
        var symbol = Visit(expr);
        var replace = shapeKind switch
        {
            CShapeKind.Shape => "shape",
            CShapeKind.Strides => "strides",
            CShapeKind.I64Dims => "dims",
            _ => throw new NotSupportedException($"Not support {shapeKind}"),
        };
        return new CSymbol(
            symbol.Type.Replace("shape", replace, StringComparison.Ordinal),
            symbol.Name.Replace("shape", replace, StringComparison.Ordinal));
    }

    private Dictionary<DimVar, DimVarInfo> CreateDimVarMap()
    {
        var varMap = new Dictionary<DimVar, DimVarInfo>(ReferenceEqualityComparer.Instance);
        var compileSession = CompileSessionScope.Current;
        if (compileSession is not null)
        {
            foreach (var (tensorVar, dimExprs) in CompileSessionScope.GetCurrentThrowIfNull().CompileOptions.ShapeBucketOptions.VarMap)
            {
                for (int i = 0; i < dimExprs.Length; i++)
                {
                    var dimExpr = dimExprs[i];
                    if (dimExpr is DimVar dimVar)
                    {
                        if (!varMap.ContainsKey(dimVar))
                        {
                            varMap.Add(dimVar, new(tensorVar, i));
                        }
                    }
                }
            }
        }

        return varMap;
    }

    private record struct DimVarInfo(IVar TensorVar, int DimIndex);
}
