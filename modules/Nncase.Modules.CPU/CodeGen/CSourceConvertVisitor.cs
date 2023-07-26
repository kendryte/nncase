// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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

internal struct IndentScope : IDisposable
{
    private static readonly AsyncLocal<IndentWriter?> _writer = new AsyncLocal<IndentWriter?>();

    private readonly bool _initialized;

    private readonly IndentWriter? _originalWriter;

    public IndentScope(StringBuilder sb)
    {
        _initialized = true;
        _writer.Value = new IndentWriter(sb);
        _originalWriter = null;
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
internal sealed class CSymbol
{
    public CSymbol(string type, string name)
    {
        Type = type;
        Name = name;
    }

    public string Type { get; }

    public string Name { get; }

    public override string ToString() => $"{Type} {Name}";
}

internal sealed class IndentWriter : StringWriter
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
internal sealed class CSourceConvertVisitor : ExprFunctor<CSymbol, Unit>
{
    private readonly Dictionary<Expr, CSymbol> _exprMemo;
    private readonly StringBuilder _implBuilder;

    public CSourceConvertVisitor()
    {
        _implBuilder = new StringBuilder();
        _exprMemo = new(ReferenceEqualityComparer.Instance);
    }

    public FunctionCSource GetFunctionCSource()
    {
        return new(_exprMemo[VisitRoot!].Type + ";", _implBuilder.ToString());
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

        var type = $"void {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(b => Visit(b).ToString()).ToArray())}, {CSourceBuiltn.FixedParameters})";

        using (var scope = new IndentScope(_implBuilder))
        {
            // 1. Function signature
            IndentScope.Writer.IndWrite($"{type} {{\n");

            // 2. Function body
            using (_ = new IndentScope())
            {
                Visit(expr.Body);
            }

            // 3. Function closing
            IndentScope.Writer.IndWrite("}\n");
        }

        symbol = new(type, new(expr.Name));
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitCall(Call expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        string type = expr.CheckedType switch
        {
            TupleType x when x == TupleType.Void => string.Empty,
            TensorType { IsScalar: true } x => x.DType.ToC(),
            _ => throw new NotSupportedException(),
        };

        string str;
        switch (expr.Target)
        {
            case IR.Math.Binary op:
                str = CSourceUtilities.ContertBinary(op, arguments);
                break;
            case IR.Math.Unary op:
                str = CSourceUtilities.ContertUnary(op, arguments);
                break;
            case Store:
                str = $"((({arguments[2].Type} *){arguments[0].Name}->vaddr)[{arguments[1].Name}] = {arguments[2].Name})";
                break;
            case Load:
                str = $"((({type} *){arguments[0].Name}->vaddr)[{arguments[1].Name}])";
                break;
            case IR.Buffers.MatchBuffer op:
                var n = arguments[0].Name;
                var pb = (TIR.PhysicalBuffer)expr[IR.Buffers.MatchBuffer.Input];
                var ind = new string(Enumerable.Repeat<char>(' ', IndentScope.Writer.Indent).ToArray());
                str = $@"uint32_t _{n}_shape[] = {{ {string.Join(", ", pb.FixedDimensions.ToArray())} }};
{ind}uint32_t _{n}_stride[] = {{ {string.Join(", ", pb.FixedStrides.ToArray())} }};
{ind}buffer_t _{n} = {{
{ind}{ind}.vaddr = ((uint8_t*) rdata + {pb.Start}),
{ind}{ind}.paddr = 0,
{ind}{ind}.shape = _{n}_shape,
{ind}{ind}.stride = _{n}_stride,
{ind}{ind}.rank = {pb.FixedDimensions.Length} }};
{ind}buffer_t *{n} = &_{n}";
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
        else
        {
            throw new NotSupportedException($"Not Support {expr.CheckedType} Const");
        }

        symbol = new(type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitVar(Var expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.CheckedType is not TensorType { Shape: { IsScalar: true } } ttype)
        {
            throw new NotSupportedException();
        }

        symbol = new(ttype.DType.ToC(), new($"{expr.Name}_{expr.GlobalVarIndex}"));
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitFor(For expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        // 1. For Loop signature
        var loopVar = Visit(expr.LoopVar);
        IndentScope.Writer.IndWrite($"for ({loopVar.Type} {loopVar.Name} = {Visit(expr.Domain.Start).Name}; {loopVar.Name} < {Visit(expr.Domain.Stop).Name}; {loopVar.Name}+={Visit(expr.Domain.Step).Name}) {{\n");
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

    /// <inheritdoc/>
    protected override CSymbol VisitSequential(Sequential expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        foreach (var field in expr.Fields)
        {
            if (field is Call call)
            {
                IndentScope.Writer.IndWrite(Visit(call).Name);
                IndentScope.Writer.Write(";\n");
            }
            else
            {
                Visit(field);
            }
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitIfThenElse(IfThenElse expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        IndentScope.Writer.IndWrite($"if({Visit(expr.Condition).Name}) {{\n");
        using (_ = new IndentScope())
        {
            Visit(expr.Then);
        }

        IndentScope.Writer.IndWrite("} else {\n");
        using (_ = new IndentScope())
        {
            Visit(expr.Else);
        }

        IndentScope.Writer.IndWrite("}\n");

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitPhysicalBuffer(PhysicalBuffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new(CSourceBuiltn.BufferType + "*", expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }
}
