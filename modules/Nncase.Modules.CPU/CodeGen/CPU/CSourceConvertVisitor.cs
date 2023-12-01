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
using Nncase.CodeGen.CPU;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.TIR;
using Razor.Templating.Core;

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
internal sealed class CSourceConvertVisitor : ExprFunctor<CSymbol, Unit>, IDisposable
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
            CSourceBuiltn.MakeKernel(ctype, _kernelBuilder.ToString()));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _sharedWriter.Dispose();
    }

    protected override CSymbol VisitVar(Var expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        symbol = new(string.Empty, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
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

        var type = $"tensor<{expr.ElemType.ToC()}, {KernelUtility.DimensionsToC(expr.Dimensions)}> ";

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
        if (expr.Target is TIR.CPU.CPUKernelOp xpuOp)
        {
            foreach (var item in expr.Arguments.ToArray().OfType<TIR.Buffer>())
            {
                DeclBuffer(item);
            }

            var args = expr.Arguments.ToArray().OfType<TIR.Buffer>().ToArray();
            switch (xpuOp)
            {
                case TIR.CPU.Unary unary:
                    IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Buffer = x, Symbol = Visit(x) }).ToArray(),
                        UnaryOp = unary.UnaryOp,
                    }).Result);
                    break;
                case TIR.CPU.TensorLoad load:
                    if (args.Length == 1)
                    {
                        var fullShape = Enumerable.Repeat(1, args[0].Dimensions.Length).ToArray();
                        var splitAxisAndScale = load.NdSbp.Select((sbp, i) => sbp is SBPSplit s ? (s.Axis, load.Placement.Hierarchy[i]) : (0, 1)).ToArray();
                        foreach (var s in splitAxisAndScale)
                        {
                            fullShape[s.Item1] *= s.Item2;
                        }

                        foreach (var (dimS, axis) in args[0].Dimensions.ToArray().Select((e, axis) => (Visit(e).Name, axis)))
                        {
                            if (int.TryParse(dimS, out var div))
                            {
                                fullShape[axis] *= div;
                            }
                            else if (CSourceUtilities.TryGetDivRem(dimS, out div, out var rem))
                            {
                                fullShape[axis] = (fullShape[axis] - 1) * div;
                                fullShape[axis] += rem;
                            }
                        }

                        IndentScope.Writer.Write($"tensor_boxing_load({Visit(args[0]).Name}, {{{string.Join(',', fullShape)}}}, {args[0].Dimensions.ToArray().Select(e => Visit(e).Name).ToSlicing(load.NdSbp, load.Placement)[1..^1]}, ctx);\n");
                    }
                    else
                    {
                        IndentScope.Writer.Write($"tensor_copy({Visit(args[1]).Name}{args[0].Dimensions.ToArray().Select(e => Visit(e).Name).ToSlicing(load.NdSbp, load.Placement)}, {Visit(args[0]).Name}.view());\n");
                    }

                    break;
                case TIR.CPU.TensorStore store:
                    if (args.Length == 1)
                    {
                        var fullShape = Enumerable.Repeat(1, args[0].Dimensions.Length).ToArray();
                        var splitAxisAndScale = store.NdSbp.Select((sbp, i) => sbp is SBPSplit s ? (s.Axis, store.Placement.Hierarchy[i]) : (0, 1)).ToArray();
                        foreach (var s in splitAxisAndScale)
                        {
                            fullShape[s.Item1] *= s.Item2;
                        }

                        foreach (var (dimS, axis) in args[0].Dimensions.ToArray().Select((e, axis) => (Visit(e).Name, axis)))
                        {
                            if (int.TryParse(dimS, out var div))
                            {
                                fullShape[axis] *= div;
                            }
                            else if (CSourceUtilities.TryGetDivRem(dimS, out div, out var rem))
                            {
                                fullShape[axis] = (fullShape[axis] - 1) * div;
                                fullShape[axis] += rem;
                            }
                        }

                        IndentScope.Writer.Write($"tensor_boxing_store({Visit(args[0]).Name}, {{{string.Join(',', fullShape)}}}, {args[0].Dimensions.ToArray().Select(e => Visit(e).Name).ToSlicing(store.NdSbp, store.Placement)[1..^1]}, ctx);\n");
                    }
                    else
                    {
                        IndentScope.Writer.Write($"tensor_copy({Visit(args[0]).Name}.view(), {Visit(args[1]).Name}{args[0].Dimensions.ToArray().Select(e => Visit(e).Name).ToSlicing(store.NdSbp, store.Placement)});\n");
                    }

                    break;
#if false
                case TIR.CPU.SwishB swishb:
                    IndentScope.Writer.Write($"swishb({Visit(args[0]).Name}, {Visit(args[1]).Name}, {swishb.Beta}, ctx)");
                    break;
                case TIR.CPU.Binary binary:
                    {
                        var ltype = (TensorType)args[0].CheckedType;
                        var rtype = (TensorType)args[1].CheckedType;
                        var outtype = (TensorType)args[2].CheckedType;
#endif
#if false
                        if (ltype.Shape.IsFixed && rtype.Shape.IsFixed)
                        {
                            var lshape = ltype.Shape;
                            var rshape = rtype.Shape;
                            var outshape = outtype.Shape;
                            var lpad = outtype.Shape.Rank - lshape.Rank;
                            var rpad = outtype.Shape.Rank - rshape.Rank;
                            var lsbp = Enumerable.Repeat<SBP>(SBP.B, binary.LhsType.Placement.Rank).ToArray();
                            var rsbp = Enumerable.Repeat<SBP>(SBP.B, binary.RhsType.Placement.Rank).ToArray();

                            var lnewShape = ltype.Shape.ToValueArray();
                            var rnewShape = rtype.Shape.ToValueArray();
                            for (int i = 0; i < binary.RhsType.Placement.Rank; i++)
                            {
                                switch (binary.LhsType.NdSBP[i], binary.RhsType.NdSBP[i])
                                {
                                    case (SBPSplit s, SBPBroadCast):
                                        var baxis = s.Axis - lshape.Rank + rshape.Rank;
                                        if (outshape[s.Axis + lpad] == lshape[s.Axis] && baxis < rshape.Rank && baxis >= 0 && lshape[s.Axis] != rshape[baxis] && rshape[baxis] != 1)
                                        {
                                            rsbp[i] = SBP.S(baxis);
                                            rnewShape[baxis] /= binary.RhsType.Placement.Hierarchy[i];
                                        }

                                        break;
                                    case (SBPBroadCast, SBPSplit s):
                                        var aaxis = s.Axis - rshape.Rank + lshape.Rank;
                                        if (outshape[s.Axis + rpad] == rshape[s.Axis] && aaxis < lshape.Rank && aaxis >= 0 && rshape[s.Axis] != lshape[aaxis] && lshape[aaxis] != 1)
                                        {
                                            lsbp[i] = SBP.S(aaxis);
                                            lnewShape[aaxis] /= binary.RhsType.Placement.Hierarchy[i];
                                        }

                                        break;
                                    default:
                                        break;
                                }
                            }


                            if (lsbp.Any(s => s is SBPSplit))
                            {
                                var slice = new Shape(lnewShape).Select(d => d.ToString()).ToSlicing(lsbp, binary.LhsType.Placement);
                                IndentScope.Writer.Write($"auto {lhsStr}_ = {lhsStr}{slice};\n");
                                lhsStr += "_";
                            }

                            if (rsbp.Any(s => s is SBPSplit))
                            {
                                var slice = new Shape(rnewShape).Select(d => d.ToString()).ToSlicing(rsbp, binary.RhsType.Placement);
                                IndentScope.Writer.Write($"auto {rhsStr}_ = {rhsStr}{slice};\n");
                                rhsStr += "_";
                            }

                        }
#endif
#if false
                    string lhsStr = Visit(args[0]).Name;
                        string rhsStr = Visit(args[1]).Name;
                        IndentScope.Writer.Write($"binary({lhsStr}, {rhsStr}, {Visit(args[2]).Name}, binary_op_t::{binary.BinaryOp.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, ctx)");
                    }

                    break;
                case TIR.CPU.Matmul matmul:
                    IndentScope.Writer.Write($"matmul({Visit(args[0]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, ctx)");
                    break;
                case TIR.CPU.LayerNorm layernorm:
                    using (_ = new IndentScope())
                    {
                        IndentScope.Writer.IndWrite($"{{\n");

                        // var dividedType = Utilities.DistributedUtility.GetDividedTensorType(layernorm.DistType);
                        IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum({{{string.Join(",", args[0].Dimensions.ToArray().Take(layernorm.Axis).Select(e => Visit(e).Name))}}});\n");
                        IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum_sqr({{{string.Join(",", args[0].Dimensions.ToArray().Take(layernorm.Axis).Select(e => Visit(e).Name))}}});\n");
                        var sbpOnAxis = layernorm.DistType.NdSBP.Where(sbp => sbp is SBPSplit s && s.Axis >= layernorm.Axis).ToArray();
                        switch (sbpOnAxis.Length)
                        {
                            case 0:
                                IndentScope.Writer.IndWrite($"reduce_sum_sqr({Visit(args[0]).Name}, sum, sum_sqr);\n");
                                IndentScope.Writer.IndWrite($"layernorm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({layernorm.Epsilon}), {layernorm.Axis}, {layernorm.DistType.TensorType.Shape[layernorm.Axis]}, {(!layernorm.UseMean).ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)});\n");
                                break;
                            case 1:
                                IndentScope.Writer.IndWrite($"reduce_sum_sqr({Visit(args[0]).Name}, sum, sum_sqr);\n");
                                if (sbpOnAxis[0] == layernorm.DistType.NdSBP[1])
                                {
                                    IndentScope.Writer.IndWrite($"tdma_thread_reduce_sync(sum, sum, reduce_op_t::sum, ctx);\n");
                                    IndentScope.Writer.IndWrite($"tdma_thread_reduce_sync(sum_sqr, sum_sqr, reduce_op_t::sum, ctx);\n");
                                }
                                else
                                {
                                    IndentScope.Writer.IndWrite($"tdma_block_reduce_sync(sum, sum, reduce_op_t::sum, ctx);\n");
                                    IndentScope.Writer.IndWrite($"tdma_block_reduce_sync(sum_sqr, sum_sqr, reduce_op_t::sum, ctx);\n");
                                }

                                IndentScope.Writer.IndWrite($"layernorm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({layernorm.Epsilon}), {layernorm.Axis}, {layernorm.DistType.TensorType.Shape[layernorm.Axis]}, {(!layernorm.UseMean).ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)});\n");
                                break;
                            case 2:
                                IndentScope.Writer.IndWrite($"reduce_sum_sqr({Visit(args[0]).Name}, sum, sum_sqr);\n");
                                IndentScope.Writer.IndWrite($"tdma_all_reduce_sync(sum, sum, reduce_op_t::sum, ctx);\n");
                                IndentScope.Writer.IndWrite($"tdma_all_reduce_sync(sum_sqr, sum_sqr, reduce_op_t::sum, ctx);\n");
                                IndentScope.Writer.IndWrite($"layernorm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({layernorm.Epsilon}), {layernorm.Axis}, {layernorm.DistType.TensorType.Shape[layernorm.Axis]}, {(!layernorm.UseMean).ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)});\n");
                                break;
                        }

                        IndentScope.Writer.IndWrite("}\n");
                    }

                    break;
                case TIR.CPU.InstanceNorm instancenorm:
                    using (_ = new IndentScope())
                    {
                        IndentScope.Writer.IndWrite($"{{\n");

                        // var dividedType = Utilities.DistributedUtility.GetDividedTensorType(instancenorm.DistType);
                        var sbpOnAxis = instancenorm.DistType.NdSBP.Where(sbp => sbp is SBPSplit s && s.Axis > 1).ToArray();
                        IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum({{{string.Join(",", args[0].Dimensions.ToArray().Take(2).Select(e => Visit(e).Name))}}});\n");
                        IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> sum_sqr({{{string.Join(",", args[0].Dimensions.ToArray().Take(2).Select(e => Visit(e).Name))}}});\n");
                        switch (sbpOnAxis.Length)
                        {
                            case 0:
                                IndentScope.Writer.IndWrite($"reduce_sum_sqr({Visit(args[0]).Name}, sum, sum_sqr);\n");
                                IndentScope.Writer.IndWrite($"instance_norm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({instancenorm.Epsilon}), {Visit(args[3].Dimensions[2]).Name});\n");
                                break;
                            case 1:
                                IndentScope.Writer.IndWrite($"reduce_sum_sqr({Visit(args[0]).Name}, sum, sum_sqr);\n");
                                if (sbpOnAxis[0] == instancenorm.DistType.NdSBP[1])
                                {
                                    IndentScope.Writer.IndWrite($"tdma_thread_reduce_sync(sum, sum, reduce_op_t::sum, ctx);\n");
                                    IndentScope.Writer.IndWrite($"tdma_thread_reduce_sync(sum_sqr, sum_sqr, reduce_op_t::sum, ctx);\n");
                                }
                                else
                                {
                                    IndentScope.Writer.IndWrite($"tdma_block_reduce_sync(sum, sum, reduce_op_t::sum, ctx);\n");
                                    IndentScope.Writer.IndWrite($"tdma_block_reduce_sync(sum_sqr, sum_sqr, reduce_op_t::sum, ctx);\n");
                                }

                                IndentScope.Writer.IndWrite($"instance_norm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({instancenorm.Epsilon}), {Visit(args[3].Dimensions[2]).Name});\n"); break;
                            case 2:
                                IndentScope.Writer.IndWrite($"tdma_all_reduce_sync(sum, sum, reduce_op_t::sum, ctx);\n");
                                IndentScope.Writer.IndWrite($"tdma_all_reduce_sync(sum_sqr, sum_sqr, reduce_op_t::sum, ctx);\n");
                                IndentScope.Writer.IndWrite($"instance_norm({Visit(args[0]).Name}, sum, sum_sqr,{Visit(args[3]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, static_cast<{args[0].CheckedDataType.ToC()}>({instancenorm.Epsilon}), {Visit(args[3].Dimensions[2]).Name});\n");
                                break;
                        }

                        IndentScope.Writer.IndWrite("}\n");
                    }

                    break;
                case TIR.CPU.Gather gather:
                    IndentScope.Writer.Write($"gather({Visit(args[0]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, {gather.Axis})");
                    break;
                case TIR.CPU.Concat concat:
                    var positiveAxis = concat.Axis > 0 ? concat.Axis : concat.Axis + ((TensorType)args[0].CheckedType).Shape.Rank;
                    IndentScope.Writer.Write($"concat({{{string.Join(",", args.SkipLast(1).Select(Visit).Select(s => "&" + s.Name))}}}, {Visit(args[^1]).Name}, {positiveAxis})");
                    break;
                case TIR.CPU.Slice slice:
                    var begins = ((TensorConst)expr.Arguments[2]).Value.ToArray<int>();
                    var ends = ((TensorConst)expr.Arguments[3]).Value.ToArray<int>();
                    var axes = ((TensorConst)expr.Arguments[4]).Value.ToArray<int>().ToList();
                    var retType = (TensorType)expr.Arguments[1].CheckedType;

                    var newbegins = Enumerable.Repeat(0, retType.Shape.Rank).ToArray();
                    var newends = retType.Shape.ToArray();
                    for (int i = 0; i < slice.DistributedType.Placement.Rank; i++)
                    {
                        var sbp = slice.DistributedType.NdSBP[i];
                        if (sbp is SBPSplit { Axis: int axis })
                        {
                            if (axes.IndexOf(axis) is int j && j != -1)
                            {
                                begins[j] /= slice.DistributedType.Placement.Hierarchy[i];
                            }
                        }
                    }

                    for (int i = 0; i < newbegins.Length; i++)
                    {
                        if (axes.IndexOf(i) is int j && j != -1)
                        {
                            newbegins[i] += begins[j];
                        }
                    }

                    IndentScope.Writer.Write($"__tensor_copy_sync(std::move({Visit(expr.Arguments[1]).Name}), {Visit(expr.Arguments[0]).Name}({{{string.Join(',', newbegins.Select(e => e.ToString()))}}},{{{string.Join(',', newends.Select(e => e.ToString()))}}}))");
                    break;
                case TIR.CPU.Softmax softmax:
                    {
                        positiveAxis = softmax.Axis > 0 ? softmax.Axis : softmax.Axis + ((TensorType)args[0].CheckedType).Shape.Rank;
                        var sbpOnAxis = softmax.DistType.NdSBP.Where(sbp => sbp is SBPSplit s && s.Axis == softmax.Axis).ToArray();
                        switch (sbpOnAxis.Length)
                        {
                            case 0:
                                IndentScope.Writer.IndWrite($"softmax({Visit(args[0]).Name}, {Visit(args[1]).Name}, {positiveAxis}, ctx, reduce_strategy_t::none)");
                                break;
                            case 1:
                                if (sbpOnAxis[0] == softmax.DistType.NdSBP[1])
                                {
                                    IndentScope.Writer.IndWrite($"softmax({Visit(args[0]).Name}, {Visit(args[1]).Name}, {positiveAxis}, ctx, reduce_strategy_t::by_thread)");
                                }
                                else
                                {
                                    IndentScope.Writer.IndWrite($"softmax({Visit(args[0]).Name}, {Visit(args[1]).Name}, {positiveAxis}, ctx, reduce_strategy_t::by_block)");
                                }

                                break;
                            case 2:
                                IndentScope.Writer.IndWrite($"softmax({Visit(args[0]).Name}, {Visit(args[1]).Name}, {positiveAxis}, ctx, reduce_strategy_t::all)");
                                break;
                        }
                    }

                    break;
                case TIR.CPU.Transpose transpose:
                    IndentScope.Writer.Write($"transpose({Visit(args[0]).Name}, {Visit(args[1]).Name}, {{{string.Join(",", transpose.Perm.Select(p => p.ToString()))}}})");
                    break;
                case TIR.CPU.ReShape reshape:
                    IndentScope.Writer.Write($"__tensor_copy_sync(std::move({Visit(args[1]).Name}),std::move(view({Visit(args[0]).Name},{{{args[1].CheckedShape.ToString()[1..^1]}}})))");
                    break;
                case TIR.CPU.GatherReduceScatter grs:
                    var ret_name = Visit(args[1]).Name;
                    bool reshard = args[0].CheckedType != args[1].CheckedType;

                    var partialSumPos = Enumerable.Range(0, grs.InType.NdSBP.Count).Where(i => grs.InType.NdSBP[i] is SBPPartialSum).Select(i => (i, grs.OutType.NdSBP[i])).ToArray();
                    var placement = grs.InType.Placement with
                    {
                        Hierarchy = new IRArray<int>(partialSumPos.Select(t => grs.InType.Placement.Hierarchy[t.i])),
                        Name = new string(partialSumPos.Select(t => grs.InType.Placement.Name[t.i]).ToArray()),
                    };

                    if (reshard)
                    {
                        IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> {ret_name}_tmp({{{string.Join(",", args[0].Dimensions.ToArray().Select(e => Visit(e).Name))}}});\n");
                    }
                    else
                    {
                        IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local>& {ret_name}_tmp = {ret_name};\n");
                    }

                    if (partialSumPos.Length == 2)
                    {
                        IndentScope.Writer.IndWrite($"tdma_all_reduce_sync({Visit(args[0]).Name}, {ret_name}_tmp, reduce_op_t::sum, ctx);\n");
                    }
                    else if (partialSumPos[0].i == 0)
                    {
                        IndentScope.Writer.IndWrite($"tdma_block_reduce_sync({Visit(args[0]).Name}, {ret_name}_tmp, reduce_op_t::sum, ctx);\n");
                    }
                    else
                    {
                        IndentScope.Writer.IndWrite($"tdma_thread_reduce_sync({Visit(args[0]).Name}, {ret_name}_tmp, reduce_op_t::sum, ctx);\n");
                    }

                    if (reshard)
                    {
                        if (!ret_name.Contains("reshape", StringComparison.CurrentCulture))
                        {
                            IndentScope.Writer.IndWrite($"__tensor_copy_sync(std::move({ret_name}), {ret_name}_tmp{args[1].Dimensions.ToArray().Select(e => Visit(e).Name).ToSlicing(new IRArray<SBP>(partialSumPos.Select(t => t.Item2)), placement)});\n");
                        }
                        else
                        {
                            var fullShape = Enumerable.Repeat(1, args[1].Dimensions.Length).ToArray();
                            var splitAxisAndScale = partialSumPos.Select((sbp, i) => sbp.Item2 is SBPSplit s ? (s.Axis, placement.Hierarchy[i]) : (0, 1)).ToArray();
                            foreach (var s in splitAxisAndScale)
                            {
                                fullShape[s.Item1] *= s.Item2;
                            }

                            foreach (var (dimS, axis) in args[1].Dimensions.ToArray().Select((e, axis) => (Visit(e).Name, axis)))
                            {
                                if (int.TryParse(dimS, out var div))
                                {
                                    fullShape[axis] *= div;
                                }
                                else if (CSourceUtilities.TryGetDivRem(dimS, out div, out var rem))
                                {
                                    fullShape[axis] = (fullShape[axis] - 1) * div;
                                    fullShape[axis] += rem;
                                }
                            }

                            IndentScope.Writer.IndWrite($"tensor<{args[0].CheckedDataType.ToC()}, loc_t::local> {ret_name}_tmp_view = view({ret_name}_tmp, {{{string.Join(',', fullShape)}}});\n");
                            IndentScope.Writer.IndWrite($"__tensor_copy_sync(std::move({ret_name}),std::move({ret_name}_tmp_view{args[1].Dimensions.ToArray().Select(e => Visit(e).Name).ToSlicing(new IRArray<SBP>(partialSumPos.Select(t => t.Item2)), placement)}));\n");
                        }
                    }

                    break;
                case TIR.CPU.Conv2D conv:
                    var sbpOnInChannel = conv.DistType.NdSBP.Where(sbp => sbp is SBPPartialSum).ToArray();
                    string strategy = sbpOnInChannel.Length switch
                    {
                        0 => "reduce_strategy_t::none",
                        1 when sbpOnInChannel[0] == conv.DistType.NdSBP[1] => "reduce_strategy_t::by_thread",
                        1 => "reduce_strategy_t::by_block",
                        2 => "reduce_strategy_t::all",
                        _ => throw new InvalidOperationException("Invalid length"),
                    };

                    IndentScope.Writer.Write($"conv2d(ctx, {Visit(args[0]).Name}, {Visit(args[1]).Name}, {Visit(args[2]).Name}, {Visit(args[3]).Name}, {{{string.Join(",", conv.Stride.Select(p => p.ToString()))}}}, {{{string.Join(",", conv.Padding.Select(p => p.ToString()))}}}, {{{string.Join(",", conv.Dilation.Select(p => p.ToString()))}}}, {conv.Groups}, {strategy})");
                    break;
                case TIR.CPU.ReduceArg reduceArg:
                    IndentScope.Writer.Write($"reduce_arg({Visit(args[0]).Name}, {Visit(args[1]).Name}, {reduceArg.Axis}, {reduceArg.KeepDims.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, {reduceArg.SelectLastIndex.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, reduce_arg_op_t::{reduceArg.ReduceArgOp.ToC()})");
                    break;
                case TIR.CPU.Resize resize:
                    using (_ = new IndentScope())
                    {
                        IndentScope.Writer.IndWrite($"{{\n");

                        IndentScope.Writer.IndWrite($"float roi[{resize.Roi.Count}] = {{{string.Join(",", resize.Roi.Select(p => p.ToString()))}}};\n");
                        IndentScope.Writer.IndWrite($"int32_t new_size[{resize.NewSize.Count}] = {{{string.Join(",", resize.NewSize.Select(p => p.ToString()))}}};\n");
                        IndentScope.Writer.IndWrite($"resize({Visit(args[0]).Name}, {Visit(args[1]).Name}, roi, new_size, {resize.CubicCoeffA.ToString()}, {resize.ExcludeOutsideValue.ToString()}, {resize.ExtrapolationValue.ToString()}, image_resize_mode_t::{resize.ResizeMode.ToC()}, image_resize_transformation_mode_t::{resize.TransformationMode.ToC()}, image_resize_nearest_mode_t::{resize.NearestMode.ToC()}, {resize.IsTFResize.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)});\n");

                        IndentScope.Writer.IndWrite("}\n");
                    }

                    break;
                case TIR.CPU.Cast cast:
                    IndentScope.Writer.Write($"cast({Visit(args[0]).Name}, {Visit(args[1]).Name})");
                    break;
                case TIR.CPU.Expand expand:
                    IndentScope.Writer.Write($"expand({Visit(args[0]).Name}, {Visit(args[1]).Name})");
                    break;
                case TIR.CPU.Clamp clamp:
                    string min = clamp.Min is float.NegativeInfinity ? float.MinValue.ToString() : clamp.Min.ToString();
                    string max = clamp.Max is float.PositiveInfinity ? float.MaxValue.ToString() : clamp.Max.ToString();
                    IndentScope.Writer.Write($"clamp({Visit(args[0]).Name}, {Visit(args[1]).Name}, (float){min}, (float){max})");
                    break;
#endif
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
                case IR.Math.Compare op:
                    str = CSourceUtilities.ContertCompare(op, arguments);
                    break;
                case IR.Math.Select op:
                    str = CSourceUtilities.ContertSelect(op, arguments);
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

        IndentScope.Writer.IndWrite($"{symbol.Type} {symbol.Name};\n");
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
