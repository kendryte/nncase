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
using Nncase.CodeGen.CPU;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.Targets;
using Nncase.TIR;
using Razor.Templating.Core;

namespace Nncase.CodeGen.CPU;

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
internal sealed class KernelCSourceConvertVisitor : ExprFunctor<CSymbol, Unit>, IDisposable
{
    private readonly Dictionary<Expr, CSymbol> _exprMemo;
    private readonly StringBuilder _kernelBuilder;

    private readonly StringBuilder _sharedBuilder;
    private readonly HashSet<TIR.PrimFunction> _refFuncs;
    private readonly StringWriter _sharedWriter;
    private ulong _collective_pool_size;

    public KernelCSourceConvertVisitor(ulong dataAlign, ulong dataUsage, ulong rdataPoolSize, ulong localRdataPoolSize, CpuTargetOptions targetOptions)
    {
        DataAlign = dataAlign;
        DataUsage = dataUsage;
        RdataPoolSize = rdataPoolSize;
        LocalRdataPoolSize = localRdataPoolSize;
        _kernelBuilder = new StringBuilder();
        _sharedBuilder = new StringBuilder();
        _sharedWriter = new StringWriter(_sharedBuilder);
        _exprMemo = new(ReferenceEqualityComparer.Instance);
        _refFuncs = new(ReferenceEqualityComparer.Instance);
        _collective_pool_size = 0;
        TargetOptions = targetOptions;
    }

    public PrimFunction VisitEntry => (TIR.PrimFunction)VisitRoot!;

    public int CallCount { get; private set; }

    public CpuTargetOptions TargetOptions { get; }

    public ulong DataAlign { get; }

    public ulong DataUsage { get; }

    public ulong RdataPoolSize { get; }

    public ulong LocalRdataPoolSize { get; }

    public KernelCSource GetCSource()
    {
        var ctype = $"void {VisitEntry.Name}({string.Join(", ", VisitEntry.Parameters.AsValueEnumerable().Select(Visit).Select(s => $"{s.Type} {s.Name}").ToArray().Concat(_exprMemo.Keys.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata).Select(Visit).Select(s => $" {s.Type} {s.Name}").ToArray()))}, uint8_t* data)";
        return new(
            CSourceBuiltn.MakeMain(VisitEntry, DataAlign, DataUsage, RdataPoolSize, LocalRdataPoolSize, _exprMemo.Keys.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata), TargetOptions),
            CSourceBuiltn.MakeKernel(ctype, _kernelBuilder.ToString()),
            CSourceBuiltn.TopoAwareRuntimeDef(TargetOptions, DataAlign, _collective_pool_size),
            CSourceBuiltn.TopologyDef(TargetOptions));
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

        var ctype = $"void {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit).Select(s => $"{s.Type} {s.Name}").ToArray())})";

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
        string loc = (expr.Location, expr.Hierarchy) switch
        {
            (MemoryLocation.Rdata, 0) => "rdata",
            (MemoryLocation.ThreadLocalRdata, 0) => "local_rdata",
            (MemoryLocation.Data, 0) => "data",
            (MemoryLocation.Data, 1) => "data",
            _ => throw new NotSupportedException($"{expr.Location}, {expr.Hierarchy}"),
        };
        var ptype = (PointerType)expr.CheckedDataType;
        var ptypeName = ptype.ElemType.ToC();
        var spanSize = ((TensorConst)expr.Size).Value.ToScalar<ulong>() / (ulong)ptype.ElemType.SizeInBytes;
        var name = $"std::span<{ptypeName}, {spanSize}> (reinterpret_cast<{ptypeName}*>({loc} + {start.Name}UL), {spanSize})";

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

        var type = VisitEntry.Parameters.AsValueEnumerable().Contains(expr) || expr.MemSpan.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata || expr.MemSpan.Start is TensorConst
            ? (expr.DistributedType == null
             ? $"tensor_view<{expr.ElemType.ToC()}, {KernelUtility.DimensionsToC(expr.Dimensions)}, {KernelUtility.StridesToC(expr.Strides)}> "
             : $"sharded_tensor_view<{expr.ElemType.ToC()}, {KernelUtility.DimensionsToC(expr.DistributedType.TensorType.Shape)}, {KernelUtility.DistributedToC(expr.DistributedType)}, {KernelUtility.StridesToC(expr.Strides)}> ")
            : $"tensor<{expr.ElemType.ToC()}, {KernelUtility.DimensionsToC(expr.Dimensions)}> ";

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
        if (expr.Target is Op kop && kop is TIR.CPU.CPUKernelOp or TIR.Memcopy)
        {
            foreach (var item in expr.Arguments.ToArray().OfType<TIR.Buffer>())
            {
                DeclBuffer(item);
            }
#if PROFILE_CALL
            IndentScope.Writer.Write($"auto start_{CallCount} = get_ms_time();\n");
#endif
            var args = expr.Arguments.ToArray().OfType<TIR.Buffer>().ToArray();
            switch (kop)
            {
                case TIR.CPU.Unary unary:
                    IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        UnaryOp = unary.UnaryOp,
                    }).Result);
                    break;
                case TIR.CPU.TensorLoad load:
                    if (args.Length == 1)
                    {
                        var fullShape = Enumerable.Repeat(1, args[0].Dimensions.Length).ToArray();
                        fullShape = fullShape.Select((l, i) => load.NdSbp[i] is SBPSplit split ? split.Axes.Select(s => load.Placement.Hierarchy[s]).Aggregate(1, (a, b) => a * b) : l).ToArray();

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

                        _collective_pool_size = Math.Max(_collective_pool_size, (ulong)(TensorUtilities.GetProduct(fullShape) * args[0].CheckedDataType.SizeInBytes));
                        var indices = args[0].Dimensions.ToArray().Select(e => Visit(e).Name).ToSlicing(load.NdSbp, load.Placement)[0];
                        IndentScope.Writer.Write($"tac::tensor_boxing_load_sync<fixed_shape<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
                    }
                    else
                    {
                        IndentScope.Writer.Write($"reshard({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: false).Name});\n");
                    }

                    break;
                case TIR.CPU.TensorStore store:
                    if (args.Length == 1)
                    {
                        var fullShape = Enumerable.Repeat(1, args[0].Dimensions.Length).ToArray();
                        fullShape = fullShape.Select((l, i) => store.NdSbp[i] is SBPSplit split ? split.Axes.Select(s => store.Placement.Hierarchy[s]).Aggregate(1, (a, b) => a * b) : l).ToArray();

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

                        _collective_pool_size = Math.Max(_collective_pool_size, (ulong)(TensorUtilities.GetProduct(fullShape) * args[0].CheckedDataType.SizeInBytes));
                        var indices = args[0].Dimensions.ToArray().Select(e => Visit(e).Name).ToSlicing(store.NdSbp, store.Placement)[0];
                        IndentScope.Writer.Write($"tac::tensor_boxing_store_sync<fixed_shape<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
                    }
                    else
                    {
                        IndentScope.Writer.Write($"reshard({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    }

                    break;
                case TIR.CPU.Binary binary:
                    {
                        IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            BinaryOp = binary.BinaryOp,
                        }).Result);
                    }

                    break;
                case TIR.CPU.Im2col im2col:
                    IndentScope.Writer.IndWrite($"im2col({VisitBuffer(args[0], local: true).Name}, fixed_shape<{string.Join(",", im2col.Kernel)}>{{}}, fixed_shape<{string.Join(",", im2col.Stride)}>{{}}, fixed_shape<{string.Join(",", im2col.Padding)}>{{}}, fixed_shape<{string.Join(",", im2col.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", im2col.PadedNums)}>{{}}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Pack pack:
                    {
                        IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.CPU.Pack>(pack)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;

                case TIR.CPU.Unpack unpack:
                    {
                        IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unpack.cshtml", new TypedKernelTemplateModel<TIR.CPU.Unpack>(unpack)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;
                case TIR.CPU.PackedLayerNorm packedLayerNorm:
                    {
                        IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedLayerNorm.cshtml", new TypedKernelTemplateModel<TIR.CPU.PackedLayerNorm>(packedLayerNorm)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.CPU.InstanceNorm instanceNorm:
                    IndentScope.Writer.Write($"instance_norm({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {args[0].ElemType.ToC()} {{ {instanceNorm.Epsilon} }}, fixed_shape<{string.Join(",", instanceNorm.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", instanceNorm.PadedNums)}>{{}} );\n");

                    break;
                case TIR.CPU.ResizeImage resize:
                    IndentScope.Writer.IndWrite($"resize({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape<{string.Join(",", resize.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", resize.PadedNums)}>{{}}, fixed_shape<{string.Join(",", resize.NewSize)}>{{}}, image_resize_mode_t::{resize.ResizeMode.ToC()}, image_resize_transformation_mode_t::{resize.TransformationMode.ToC()}, image_resize_nearest_mode_t::{resize.NearestMode.ToC()});\n");
                    break;
                case TIR.CPU.PackedSoftmax packedsoftmax:
                    {
                        IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedSoftMax.cshtml", new TypedKernelTemplateModel<TIR.CPU.PackedSoftmax>(packedsoftmax)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.CPU.PackedBinary packedBinary:
                    {
                        IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                        {
                            BinaryOp = packedBinary.BinaryOp,
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.CPU.Conv2D conv:
                    IndentScope.Writer.IndWrite($"conv2d({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, fixed_shape<{string.Join(",", conv.Stride)}>{{}}, fixed_shape<{string.Join(",", conv.Padding)}>{{}}, fixed_shape<{string.Join(",", conv.Dilation)}>{{}}, {conv.Groups});\n");
                    break;
                case TIR.CPU.Matmul matmul:
                    IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.CPU.Matmul>(matmul)
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                    }).Result);

                    break;
                case TIR.CPU.SUMMA summa:
                    IndentScope.Writer.IndWrite($"summa<false>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name}, {VisitBuffer(args[2], local: false).Name});\n");
                    break;
                case TIR.Memcopy copy:
                    IndentScope.Writer.Write($"tensor_copy({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: true).Name});\n");
                    break;
                case TIR.CPU.Gather gather:
                    IndentScope.Writer.Write($"gather<{gather.Axis}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.CPU.Swish swish:
                    if (swish.Beta != 1.0f)
                    {
                        throw new NotSupportedException();
                    }

                    IndentScope.Writer.Write($"unary<ops::swish>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Slice slice:
                    IndentScope.Writer.Write($"slice<fixed_shape<{string.Join(",", slice.Begins)}>, fixed_shape<{string.Join(",", slice.Ends)}>, fixed_shape<{string.Join(",", slice.Axes)}>, fixed_shape<{string.Join(",", slice.Strides)}>>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Concat concat:
                    IndentScope.Writer.Write($"concat<{concat.Axis}>(std::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name});\n");
                    break;
                case TIR.CPU.Transpose transpose:
                    IndentScope.Writer.Write($"transpose<fixed_shape<{string.Join(",", transpose.Perm)}>>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Pad pad:
                    IndentScope.Writer.Write($"pad<{string.Join(",", pad.Paddings)}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {args[0].CheckedDataType.ToC()} {{ {pad.PadValue} }} );\n");
                    break;
                case TIR.CPU.Reduce reduce:
                    IndentScope.Writer.Write($"reduce_{reduce.ReduceOp.ToC()}<fixed_shape<{string.Join(",", reduce.Axes)}>, fixed_shape<{string.Join(",", reduce.PackedAxes)}>, fixed_shape<{string.Join(",", reduce.PadedNums)}>>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.ReduceArg reduceArg:
                    IndentScope.Writer.Write($"reduce_arg<ops::{reduceArg.ReduceArgOp.ToC()[4..]}, {reduceArg.Axis}, {reduceArg.SelectLastIndex.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, {reduceArg.KeepDims.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape<>{{}}, fixed_shape<>{{}});\n");
                    break;
                case TIR.CPU.Clamp clamp:
                    string min = clamp.Min is float.NegativeInfinity ? float.MinValue.ToString() : clamp.Min.ToString();
                    string max = clamp.Max is float.PositiveInfinity ? float.MaxValue.ToString() : clamp.Max.ToString();
                    IndentScope.Writer.Write($"clamp({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, (float){min}, (float){max});\n");
                    break;
                case TIR.CPU.Cast cast:
                    IndentScope.Writer.Write($"cast({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Where where:
                    IndentScope.Writer.Write($"where({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.CPU.Expand expand:
                    IndentScope.Writer.Write($"expand({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Erf erf:
                    IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        UnaryOp = UnaryOp.Erf,
                    }).Result);
                    break;
                case TIR.CPU.Compare compare:
                    {
                        IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Compare.cshtml", new CompareKernelTemplateModel
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            CompareOp = compare.CompareOp,
                        }).Result);
                    }

                    break;
                case TIR.CPU.ScatterND scatterND:
                    IndentScope.Writer.Write($"scatter_nd({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");

                    break;
                case TIR.CPU.GatherReduceScatter grs:
                    {
                        if (grs.InType.AxisPolices.Any(s => s is SBPPartial))
                        {
                            var sbpPartial = (SBPPartial)grs.InType.AxisPolices.Where(s => s is SBPPartial).Distinct().First();
                            var reduceKind = "tar::reduce_kind::" + string.Join("_", grs.InType.AxisPolices.Select((s, i) => (s is SBPPartial ? "r" : string.Empty) + TargetOptions.HierarchyNames[i]));
                            IndentScope.Writer.IndWrite($"tac::tensor_reduce_sync<reduce_op::{sbpPartial.Op.ToC()}, {reduceKind}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                        }
                        else
                        {
                            var fullShape = Enumerable.Repeat(1, args[0].Dimensions.Length).ToArray();
                            fullShape = fullShape.Select((l, i) => grs.InType.AxisPolices[i] is SBPSplit split ? split.Axes.Select(s => grs.InType.Placement.Hierarchy[s]).Aggregate(1, (a, b) => a * b) : l).ToArray();

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

                            _collective_pool_size = Math.Max(_collective_pool_size, (ulong)(TensorUtilities.GetProduct(fullShape) * args[0].CheckedDataType.SizeInBytes));
                            IndentScope.Writer.Write($"reshard({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name});\n");
                        }
                    }

                    break;
                case TIR.CPU.Reshape reshape:
                    throw new NotSupportedException("the reshape should be eliminated");
                default:
                    throw new NotSupportedException(kop.ToString());
            }
#if PROFILE_CALL
            IndentScope.Writer.Write($"printf(\"{expr.Target.GetType().Name} cost: %f\\n\", get_ms_time() - start_{CallCount++});\n");
#endif
        }
        else if (expr.Target is PrimFunction deviceFunc)
        {
            foreach (var item in expr.Arguments.ToArray().OfType<TIR.Buffer>())
            {
                DeclBuffer(item);
            }
#if DEBUG_PRINT
            IndentScope.Writer.IndWrite($"runtime_util->printf(\"call {deviceFunc.Name} bid %d tid %d\\n\", bid, tid);\n");
#endif
            var arguments = expr.Arguments.AsValueEnumerable().Select(x => x switch
            {
                TIR.Buffer b => VisitBuffer(b, local: true),
                _ => Visit(x),
            }).ToArray();
            _refFuncs.Add(deviceFunc);
            IndentScope.Writer.IndWrite($"{deviceFunc.Name}({string.Join(",", arguments.Select(arg => arg.Name))});\n");
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
                case TIR.Load op:
                    str = $"{arguments[0].Name}[{arguments[1].Name}]";
                    break;
                case TIR.Store op:
                    IndentScope.Writer.IndWrite($"{arguments[0].Name}[{arguments[1].Name}] = {arguments[1].Name};\n");
                    break;
                case TIR.CPU.PtrOf op:
                    str = op.PtrName;
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
        else if (expr is TensorConst { Value: Tensor { ElementType: PointerType { ElemType: DataType }, Shape: { IsScalar: true } } pointer })
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

    private CSymbol VisitBuffer(TIR.Buffer buffer, bool local)
    {
        var symbol = VisitBuffer(buffer);
        if (local && buffer.DistributedType != null)
        {
            return new CSymbol(symbol.Type, $"{symbol.Name}.local()");
        }

        return symbol;
    }

    private void DeclBuffer(TIR.Buffer buffer)
    {
        if (_exprMemo.ContainsKey(buffer))
        {
            return;
        }

        var symbol = Visit(buffer);

        if (buffer.MemSpan.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata)
        {
            return;
        }

        IndentScope.Writer.IndWrite($"{symbol.Type} {symbol.Name}");
        if (buffer.MemSpan.Start is not None)
        {
            IndentScope.Writer.IndWrite($"({Visit(buffer.MemSpan).Name})");
        }

        IndentScope.Writer.Write($";\n");
    }
}
