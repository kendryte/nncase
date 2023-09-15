// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define MULTI_CORE_CPU

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
internal sealed class CSymbol
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

#if MULTI_CORE_CPU
/// <summary>
/// convert single prim function to c source.
/// </summary>
internal sealed class CSourceConvertVisitor : ExprFunctor<CSymbol, Unit>
{
    private readonly Dictionary<Expr, CSymbol> _exprMemo;
    private readonly StringBuilder _kernelBuilder;

    private readonly StringBuilder _sharedBuilder;
    private readonly StringWriter _sharedWriter;

    public CSourceConvertVisitor()
    {
        _kernelBuilder = new StringBuilder();
        _sharedBuilder = new StringBuilder();
        _sharedWriter = new StringWriter(_sharedBuilder);
        _exprMemo = new(ReferenceEqualityComparer.Instance);
    }

    public PrimFunction VisitEntry => (TIR.PrimFunction)VisitRoot!;

    public FunctionCSource GetFunctionCSource()
    {
        _exprMemo.Keys.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location == MemoryLocation.L2Data).ToList().ForEach(b =>
        {
            _sharedWriter.Write(_exprMemo[b]);
            _sharedWriter.WriteLine(";");
        });

        var ctype = $"void {VisitEntry.Name}({string.Join(", ", VisitEntry.Parameters.AsValueEnumerable().Select(Visit).Select(s => $"{s.Type} &{s.Name}").ToArray().Concat(_exprMemo.Keys.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location == MemoryLocation.Rdata).Select(Visit).Select(s => $" {s.Type} &{s.Name}").ToArray()))})";
        return new(
            CSourceBuiltn.MakeMain(VisitEntry, _exprMemo.Keys.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location == MemoryLocation.Rdata)),
            CSourceBuiltn.MakeShared(_sharedBuilder.ToString()),
            CSourceBuiltn.MakeKernel(ctype, _kernelBuilder.ToString()));
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

        using (var scope = new IndentScope(_kernelBuilder))
        {
            // 1. Function signature
            IndentScope.Writer.IndWrite($"{{\n");

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

    /// <inheritdoc/>
    protected override CSymbol VisitMemSpan(MemSpan expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var start = Visit(expr.Start);
        _ = Visit(expr.Size);
        string name = start.Name;
        if (expr.Start is TensorConst or Call)
        {
            var loc = expr.Location switch
            {
                MemoryLocation.Rdata => "rdata",
                MemoryLocation.Data => "data",
                _ => throw new NotSupportedException(),
            };
            name = $"({loc} + {start.Name})";
        }

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

        var type = $"tensor<{expr.ElemType.ToC()}, {expr.MemSpan.Location.ToC()}> ";

        symbol = new(type, expr.Name);
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

        string type = expr.CheckedType switch
        {
            TupleType x when x == TupleType.Void => string.Empty,
            TensorType { IsScalar: true } x => x.DType.ToC(),
            _ => throw new NotSupportedException(),
        };

        string str = string.Empty;
        if (expr.Target is IR.XPU.XPUKernelOp xpuOp)
        {
            foreach (var item in expr.Arguments.ToArray().OfType<TIR.Buffer>())
            {
                DeclBuffer(item);
            }

            var args = expr.Arguments.ToArray().OfType<TIR.Buffer>().ToArray();
            switch (xpuOp)
            {
                case IR.XPU.TDMALoad load:
                    IndentScope.Writer.Write($"tdma_load_async({Visit(args[0]).Name}, {Visit(args[1]).Name}{((TensorType)args[0].CheckedType).ToSlicing(load.NdSbp, load.Placement)})");
                    break;
                case IR.XPU.TDMAStore store:
                    IndentScope.Writer.Write($"tdma_store_async({Visit(args[0]).Name}, {Visit(args[1]).Name}{((TensorType)args[0].CheckedType).ToSlicing(store.NdSbp, store.Placement)})");
                    break;
                case IR.XPU.Unary unary:
                    IndentScope.Writer.Write($"unary({Visit(args[0]).Name}, {Visit(args[1]).Name}, unary_op_t::{unary.UnaryOp.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)})");
                    break;
                case IR.XPU.Binary binary:
                    IndentScope.Writer.Write($"binary({Visit(args[0]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, binary_op_t::{binary.BinaryOp.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)})");
                    break;
                case IR.XPU.Matmul matmul:
                    IndentScope.Writer.Write($"matmul({Visit(args[0]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name})");
                    break;
                case IR.XPU.LayerNorm layernorm:
                    using (_ = new IndentScope())
                    {
                        IndentScope.Writer.IndWrite($"{{\n");

                        var dividedType = DistributedUtilities.GetDividedTensorType(layernorm.DistType);
                        var sbpOnAxis = layernorm.DistType.NdSbp.Where(sbp => sbp is SBPSplit s && s.Axis >= layernorm.Axis).ToArray();
                        switch (sbpOnAxis.Length)
                        {
                            case 0:
                                IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum({{{string.Join(",", dividedType.Shape.ToArray().Take(layernorm.Axis))}}});\n");
                                IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum_sqr({{{string.Join(",", dividedType.Shape.ToArray().Take(layernorm.Axis))}}});\n");
                                IndentScope.Writer.IndWrite($"reduce_sum_sqr({Visit(args[0]).Name}, sum, sum_sqr);\n");
                                IndentScope.Writer.IndWrite($"layernorm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({layernorm.Epsilon}), {layernorm.Axis}, {layernorm.DistType.TensorType.Shape[layernorm.Axis]}, {(!layernorm.UseMean).ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)});\n");
                                break;
                            case 1:
                                IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum({{{string.Join(",", dividedType.Shape.ToArray().Take(layernorm.Axis))}}});\n");
                                IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum_sqr({{{string.Join(",", dividedType.Shape.ToArray().Take(layernorm.Axis))}}});\n");
                                IndentScope.Writer.IndWrite($"reduce_sum_sqr({Visit(args[0]).Name}, sum, sum_sqr);\n");
                                if (sbpOnAxis[0] == layernorm.DistType.NdSbp[1])
                                {
                                    IndentScope.Writer.IndWrite($"tdma_reduce_async(sum, sum, reduce_op_t::sum, ctx);\n");
                                    IndentScope.Writer.IndWrite($"tdma_reduce_async(sum_sqr, sum_sqr, reduce_op_t::sum, ctx);\n");
                                }
                                else
                                {
                                    IndentScope.Writer.IndWrite($"tdma_all_reduce_async(sum, sum, reduce_op_t::sum, reduce_strategy_t::by_block, ctx);\n");
                                    IndentScope.Writer.IndWrite($"tdma_all_reduce_async(sum_sqr, sum_sqr, reduce_op_t::sum, reduce_strategy_t::by_block, ctx);\n");
                                }

                                IndentScope.Writer.IndWrite($"layernorm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({layernorm.Epsilon}), {layernorm.Axis}, {layernorm.DistType.TensorType.Shape[layernorm.Axis]}, {(!layernorm.UseMean).ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)});\n"); break;
                            case 2:
                                IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum({{{string.Join(",", dividedType.Shape.ToArray().Take(layernorm.Axis))}}});\n");
                                IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum_sqr({{{string.Join(",", dividedType.Shape.ToArray().Take(layernorm.Axis))}}});\n");
                                IndentScope.Writer.IndWrite($"reduce_sum_sqr({Visit(args[0]).Name}, sum, sum_sqr);\n");
                                IndentScope.Writer.IndWrite($"tdma_all_reduce_async(sum, sum, reduce_op_t::sum, reduce_strategy_t::all, ctx);\n");
                                IndentScope.Writer.IndWrite($"tdma_all_reduce_async(sum_sqr, sum_sqr, reduce_op_t::sum, reduce_strategy_t::all, ctx);\n");
                                IndentScope.Writer.IndWrite($"layernorm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({layernorm.Epsilon}), {layernorm.Axis}, {layernorm.DistType.TensorType.Shape[layernorm.Axis]}, {(!layernorm.UseMean).ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)});\n");
                                break;
                        }

                        IndentScope.Writer.IndWrite("}\n");
                    }

                    break;
                default:
                    throw new NotSupportedException(xpuOp.ToString());
            }
        }
        else
        {
            var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
            switch (expr.Target)
            {
                case IR.Math.Binary op:
                    str = CSourceUtilities.ContertBinary(op, arguments);
                    break;
                case IR.Math.Unary op:
                    str = CSourceUtilities.ContertUnary(op, arguments);
                    break;
                default:
                    throw new NotSupportedException();
            }
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
            type = "uint8_t *";
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

    private void DeclBuffer(TIR.Buffer buffer)
    {
        if (_exprMemo.ContainsKey(buffer))
        {
            return;
        }

        var symbol = Visit(buffer);

        if (buffer.MemSpan.Location == MemoryLocation.Rdata)
        {
            return;
        }

        IndentScope.Writer.IndWrite($"{symbol.Type} {symbol.Name}({{{string.Join(',', buffer.Dimensions.AsValueEnumerable().Select(Visit).Select(s => s.Name).ToArray())}}});\n");
    }
}
#else
/// <summary>
/// convert single prim function to c source.
/// </summary>
internal sealed class CSourceConvertVisitor : ExprFunctor<CSymbol, Unit>
{
    private readonly Dictionary<Expr, CSymbol> _exprMemo;
    private readonly StringBuilder _implBuilder;
    private readonly StringBuilder _declBuilder;
    private readonly StringWriter _declWriter;

    public CSourceConvertVisitor()
    {
        _implBuilder = new StringBuilder();
        _declBuilder = new StringBuilder();
        _declWriter = new StringWriter(_declBuilder);
        _exprMemo = new(ReferenceEqualityComparer.Instance);
    }

    public PrimFunction VisitEntry => (TIR.PrimFunction)VisitRoot!;

    public FunctionCSource GetFunctionCSource()
    {
        return new(_declBuilder.ToString(), _implBuilder.ToString());
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

        var type = $"void {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(b => Visit(b.MemSpan.Start).ToString()).ToArray())}, {CSourceBuiltn.FixedParameters})";

        _declWriter.WriteLine(type + ";");
        _declWriter.WriteLine();

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

        var ctype = $"void (*{expr.Name})({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(b => Visit(b.MemSpan.Start).ToString()).ToArray())}, {CSourceBuiltn.FixedParameters})";
        symbol = new(ctype, expr.Name);
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
        string name = start.Name;
        if (expr.Start is TensorConst or Call)
        {
            var loc = expr.Location switch
            {
                MemoryLocation.Rdata => "rdata",
                MemoryLocation.Data => "data",
                _ => throw new NotSupportedException(),
            };
            name = $"({loc} + {start.Name})";
        }

        symbol = new(start.Type, name);
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
                str = $"((({arguments[2].Type} *){arguments[0].Name})[{arguments[1].Name}] = {arguments[2].Name})";
                break;
            case Load:
                str = $"((({type} *){arguments[0].Name})[{arguments[1].Name}])";
                break;
            case IR.Buffers.MatchBuffer op:
                var n = arguments[0].Name;
                var pb = (TIR.Buffer)expr[IR.Buffers.MatchBuffer.Input];
                var ind = new string(Enumerable.Repeat<char>(' ', IndentScope.Writer.Indent).ToArray());
                str = $@"uint32_t _{n}_shape[] = {{ {string.Join(", ", pb.Dimensions.AsValueEnumerable().Select(e => Visit(e).Name).ToArray())} }};
{ind}uint32_t _{n}_stride[] = {{ {string.Join(", ", pb.Strides.AsValueEnumerable().Select(e => Visit(e).Name).ToArray())} }};
{ind}buffer_t _{n} = {{
{ind}{ind}.vaddr = ((uint8_t*) rdata + {Visit(pb.MemSpan.Start).Name}),
{ind}{ind}.paddr = 0,
{ind}{ind}.shape = _{n}_shape,
{ind}{ind}.stride = _{n}_stride,
{ind}{ind}.rank = {pb.Dimensions.Length} }};
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
        else if (expr is TensorConst { Value: Tensor { ElementType: PointerType { ElemType: PrimType }, Shape: { IsScalar: true } } pointer })
        {
            str = pointer.ToScalar<ulong>().ToString();
            type = "uint8_t *";
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

        if (expr.Mode == LoopMode.Parallel)
        {
            // find the vars will be used and make new struct type.
            var msg_fields = _exprMemo.Where(p => p.Key is MemSpan or TIR.Buffer or Var).Select(p => p.Value).Concat(CSymbol.Builtns).ToArray();
            var msg_type = DeclThreadMessageStruct(msg_fields);

            using (new IndentScope(_declBuilder))
            {
                IndentScope.Writer.IndWrite($"void *{VisitEntry.Name}_inner(void *args) {{\n");
                using (new IndentScope())
                {
                    IndentScope.Writer.IndWrite($"{msg_type}* _message = ({msg_type}*)args;\n");
                    foreach (var sym in msg_fields)
                    {
                        IndentScope.Writer.IndWrite($"{sym.Type} {sym.Name} = _message->{sym.Name};\n");
                    }

                    Visit(expr.Body);
                }

                IndentScope.Writer.IndWrite(" return 0;\n");
                IndentScope.Writer.IndWrite("}\n");
            }

            using (new IndentScope())
            {
                IndentScope.Writer.IndWrite($"{msg_type} _message = {{\n");
                foreach (var sym in msg_fields)
                {
                    IndentScope.Writer.IndWrite($".{sym.Name} = {sym.Name},\n");
                }

                IndentScope.Writer.IndWrite("};\n");

                IndentScope.Writer.IndWrite($"nncase_mt->thread_start({VisitEntry.Name}_inner, (void *)&_message, sizeof ({msg_type}));\n");
            }
        }
        else
        {
            using (_ = new IndentScope())
            {
                // 2. For Body
                Visit(expr.Body);
            }
        }

        // 3. For closing
        IndentScope.Writer.IndWrite("}\n");

        if (expr.Mode == LoopMode.Parallel)
        {
            IndentScope.Writer.IndWrite("nncase_mt->thread_end();\n");
        }

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

    private string DeclThreadMessageStruct(IEnumerable<CSymbol> keyValues)
    {
        var type = $"{VisitEntry.Name}_thread_message_t";
        _declWriter.WriteLine("typedef struct {");
        foreach (var sym in keyValues)
        {
            if (sym.Name == string.Empty)
            {
                throw new InvalidOperationException("empty name");
            }

            _declWriter.WriteLine("  " + sym.Type + " " + sym.Name + ";");
        }

        _declWriter.WriteLine($"}} {type};");
        _declWriter.WriteLine();
        return type;
    }
}
#endif
