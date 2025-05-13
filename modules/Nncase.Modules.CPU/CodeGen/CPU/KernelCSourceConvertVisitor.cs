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

    public static void WriteWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal); // 找到第一个 '(' 的位置
            if (index != -1)
            {
                tagName = functionName.Substring(0, index); // 截取从头到 '(' 之前的部分
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
        IndentScope.Writer.Write($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.Write($"auto_profiler profiler(function_name, runtime::profiling_level::kernel);\n");
        IndentScope.Writer.Write($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public static void WriteIndWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal); // 找到第一个 '(' 的位置
            if (index != -1)
            {
                tagName = functionName.Substring(0, index); // 截取从头到 '(' 之前的部分
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
        IndentScope.Writer.IndWrite($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.IndWrite($"auto_profiler profiler(function_name, runtime::profiling_level::kernel);\n");
        IndentScope.Writer.IndWrite($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public KernelCSource GetCSource()
    {
        var ctype = $"template<{string.Join(", ", Enumerable.Range(0, VisitEntry.Parameters.Length).Select(x => $"class T{x}"))}>" + Environment.NewLine +
            $"void {VisitEntry.Name}({string.Join(", ", VisitEntry.Parameters.AsValueEnumerable().Select(Visit).Select(s => $"{s.Type} {s.Name}").ToArray().Concat(_exprMemo.Keys.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata).Select(Visit).Select(s => $" {s.Type} {s.Name}").ToArray()))}, uint8_t* data)";
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

        var name = IRHelpers.GetIdentityName(expr.Name);
        var index = VisitEntry.Parameters.IndexOf(expr);
        if (index != -1)
        {
            symbol = new CSymbol($"T{index}", name);
        }
        else
        {
            symbol = new(expr.CheckedDataType.ToC(), name);
        }

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

        var dimensions = expr.DistributedType is null ? expr.Dimensions : expr.DistributedType.TensorType.Shape.Dimensions;
        var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x is TensorConst);
        var isFixedStrides = expr.Strides.AsValueEnumerable().All(x => x is TensorConst);
        var dimensionSymbols = dimensions.AsValueEnumerable().Select(Visit).ToArray();
        var strideSymbols = expr.Strides.AsValueEnumerable().Select(Visit).ToArray();

        var dtypeStr = expr.ElemType.ToC();
        var dimensionStr = KernelUtility.DimensionsToC(isFixedDimensions, dimensionSymbols, true);
        var strideStr = KernelUtility.StridesToC(isFixedStrides, strideSymbols, true);

        var type = expr.MemSpan.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata || expr.MemSpan.Start is TensorConst
            ? (expr.DistributedType == null
             ? $"tensor_view<{dtypeStr}, {dimensionStr}, {strideStr}> "
             : $"sharded_tensor_view<{dtypeStr}, {dimensionStr}, {KernelUtility.DistributedToC(expr.DistributedType)}, {strideStr}> ")
            : $"tensor<{dtypeStr}, {dimensionStr}, {strideStr}> ";

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
            var args = expr.Arguments.ToArray();
            switch (kop)
            {
                case TIR.CPU.Unary unary:
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        UnaryOp = unary.UnaryOp,
                    }).Result);
                    break;
                case TIR.CPU.TensorLoad load:
                    if (args.Length == 1)
                    {
                        var fullShape = args[0].CheckedShape.ToValueArray();
                        (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                        _collective_pool_size = Math.Max(_collective_pool_size, (ulong)maxSize);
                        var indices = args[0].CheckedShape.Select(e => Visit(e.Value).Name).ToSlicing(load.NdSbp, load.Placement)[0];
                        WriteWithProfiler($"tac::tensor_boxing_load_sync<fixed_shape<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
                    }
                    else
                    {
                        WriteWithProfiler($"reshard({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: false).Name});\n");
                    }

                    break;
                case TIR.CPU.TensorStore store:
                    if (args.Length == 1)
                    {
                        var fullShape = args[0].CheckedShape.ToValueArray();
                        (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                        _collective_pool_size = Math.Max(_collective_pool_size, (ulong)maxSize);
                        var indices = args[0].CheckedShape.Select(e => Visit(e.Value).Name).ToSlicing(store.NdSbp, store.Placement)[0];
                        WriteWithProfiler($"tac::tensor_boxing_store_sync<fixed_shape<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
                    }
                    else
                    {
                        WriteWithProfiler($"reshard({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    }

                    break;
                case TIR.CPU.Binary binary:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            BinaryOp = binary.BinaryOp,
                        }).Result);
                    }

                    break;
                case TIR.CPU.Im2col im2col:
                    WriteIndWithProfiler($"im2col({VisitBuffer(args[0], local: true).Name}, fixed_shape<{string.Join(",", im2col.Kernel)}>{{}}, fixed_shape<{string.Join(",", im2col.Stride)}>{{}}, fixed_shape<{string.Join(",", im2col.Padding)}>{{}}, fixed_shape<{string.Join(",", im2col.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", im2col.PadedNums)}>{{}}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Pack pack:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.CPU.Pack>(pack)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;

                case TIR.CPU.Unpack unpack:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unpack.cshtml", new TypedKernelTemplateModel<TIR.CPU.Unpack>(unpack)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;
                case TIR.CPU.PackedLayerNorm packedLayerNorm:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedLayerNorm.cshtml", new TypedKernelTemplateModel<TIR.CPU.PackedLayerNorm>(packedLayerNorm)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.CPU.InstanceNorm instanceNorm:
                    WriteWithProfiler($"instance_norm({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {args[0].CheckedDataType.ToC()} {{ {instanceNorm.Epsilon} }}, fixed_shape<{string.Join(",", instanceNorm.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", instanceNorm.PadedNums)}>{{}} );\n");
                    break;
                case TIR.CPU.ResizeImage resize:
                    WriteIndWithProfiler($"resize({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape<{string.Join(",", resize.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", resize.PadedNums)}>{{}}, fixed_shape<{string.Join(",", resize.NewSize)}>{{}}, image_resize_mode_t::{resize.ResizeMode.ToC()}, image_resize_transformation_mode_t::{resize.TransformationMode.ToC()}, image_resize_nearest_mode_t::{resize.NearestMode.ToC()});\n");
                    break;
                case TIR.CPU.PackedSoftmax packedsoftmax:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedSoftMax.cshtml", new TypedKernelTemplateModel<TIR.CPU.PackedSoftmax>(packedsoftmax)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.CPU.PackedBinary packedBinary:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                        {
                            BinaryOp = packedBinary.BinaryOp,
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.CPU.Conv2D conv:
                    WriteIndWithProfiler($"conv2d({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, fixed_shape<{string.Join(",", conv.Stride)}>{{}}, fixed_shape<{string.Join(",", conv.Padding)}>{{}}, fixed_shape<{string.Join(",", conv.Dilation)}>{{}}, {conv.Groups});\n");
                    break;
                case TIR.CPU.Matmul matmul:
                    IndentScope.Writer.IndWrite(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.CPU.Matmul>(matmul)
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                    }).Result);

                    break;
                case TIR.CPU.SUMMA summa:
                    var rdKind = "tar::reduce_kind::" + string.Join("_", Enumerable.Range(0, TargetOptions.HierarchyNames.Length).Select(i => i >= TargetOptions.HierarchyNames.Length - 2 ? "r" + TargetOptions.HierarchyNames[i] : string.Empty + TargetOptions.HierarchyNames[i]));
                    IndentScope.Writer.IndWrite($"{{tac::detail::tensor_reduce_sync_impl<reduce_op::sum, {rdKind}> impl; impl.reduce_group_sync();\n");
                    IndentScope.Writer.IndWrite($"summa<false>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name}, {VisitBuffer(args[2], local: false).Name}, fixed_shape<{string.Join(",", summa.LhsPackedAxes)}>{{}}, fixed_shape<{string.Join(",", summa.LhsPadedNums)}>{{}}, fixed_shape<{string.Join(",", summa.RhsPackedAxes)}>{{}}, fixed_shape<{string.Join(",", summa.RhsPadedNums)}>{{}});\n");
                    IndentScope.Writer.IndWrite($"impl.reduce_group_sync();}}\n");
                    break;
                case TIR.Memcopy copy:
                    WriteWithProfiler($"tensor_copy({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: true).Name});\n");
                    break;
                case TIR.CPU.Gather gather:
                    WriteWithProfiler($"gather<{gather.Axis}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.CPU.Swish swish:
                    if (swish.Beta != 1.0f)
                    {
                        throw new NotSupportedException();
                    }

                    WriteWithProfiler($"unary<ops::swish>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Slice slice:
                    WriteWithProfiler($"slice<fixed_dims<int64_t, {string.Join(",", slice.Axes)}>, fixed_dims<int64_t, {string.Join(",", slice.Strides)}>>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.CPU.Concat concat:
                    WriteWithProfiler($"concat<{concat.Axis}>(std::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name});\n");
                    break;
                case TIR.CPU.Transpose transpose:
                    WriteWithProfiler($"transpose<fixed_shape<{string.Join(",", transpose.Perm)}>>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Pad pad:
                    WriteWithProfiler($"pad<{string.Join(",", pad.Paddings)}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {args[0].CheckedDataType.ToC()} {{ {pad.PadValue} }} );\n");
                    break;
                case TIR.CPU.Reduce reduce:
                    WriteWithProfiler($"reduce_{reduce.ReduceOp.ToC()}<fixed_shape<{string.Join(",", reduce.Axes)}>, fixed_shape<{string.Join(",", reduce.PackedAxes)}>, fixed_shape<{string.Join(",", reduce.PadedNums)}>>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.ReduceArg reduceArg:
                    WriteWithProfiler($"reduce_arg<ops::{reduceArg.ReduceArgOp.ToC()[4..]}, {reduceArg.Axis}, {reduceArg.SelectLastIndex.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, {reduceArg.KeepDims.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape<>{{}}, fixed_shape<>{{}});\n");
                    break;
                case TIR.CPU.Clamp clamp:
                    string min = clamp.Min is float.NegativeInfinity ? float.MinValue.ToString() : clamp.Min.ToString();
                    string max = clamp.Max is float.PositiveInfinity ? float.MaxValue.ToString() : clamp.Max.ToString();
                    WriteWithProfiler($"clamp({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, (float){min}, (float){max});\n");
                    break;
                case TIR.CPU.Cast cast:
                    WriteWithProfiler($"cast({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Where where:
                    WriteWithProfiler($"where({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.CPU.Expand expand:
                    WriteWithProfiler($"expand({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.Erf erf:
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        UnaryOp = UnaryOp.Erf,
                    }).Result);
                    break;
                case TIR.CPU.Compare compare:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Compare.cshtml", new CompareKernelTemplateModel
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            CompareOp = compare.CompareOp,
                        }).Result);
                    }

                    break;
                case TIR.CPU.ScatterND scatterND:
                    WriteWithProfiler($"scatter_nd({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");

                    break;
                case TIR.CPU.GatherReduceScatter grs:
                    {
                        if (grs.InType.AxisPolices.Any(s => s is SBPPartial))
                        {
                            var sbpPartial = (SBPPartial)grs.InType.AxisPolices.Where(s => s is SBPPartial).Distinct().First();
                            var reduceKind = "tar::reduce_kind::" + string.Join("_", grs.InType.AxisPolices.Select((s, i) => (s is SBPPartial ? "r" : string.Empty) + TargetOptions.HierarchyNames[i]));
                            WriteIndWithProfiler($"tac::tensor_reduce_sync<reduce_op::{sbpPartial.Op.ToC()}, {reduceKind}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                        }
                        else
                        {
                            (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                            _collective_pool_size = Math.Max(_collective_pool_size, (ulong)maxSize);
                            WriteWithProfiler($"reshard({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name});\n");
                        }
                    }

                    break;
                case TIR.CPU.GetItem getItem:
                    IndentScope.Writer.Write($"get_item({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.CPU.Stack stack:
                    IndentScope.Writer.Write($"stack<{stack.Axis}>(std::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name});\n");
                    break;
                case TIR.CPU.Reshape reshape:
                    IndentScope.Writer.Write($"reshape({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.CPU.UpdatePagedAttentionKVCache updatePagedAttentionKVCache:
                    IndentScope.Writer.IndWrite($"update_paged_attention_kv_cache<{updatePagedAttentionKVCache.Layout.ToC()}>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, caching::attention_cache_kind::{updatePagedAttentionKVCache.CacheKind.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, {updatePagedAttentionKVCache.LayerId});\n");
                    break;
                case TIR.CPU.GatherPagedAttentionKVCache gakv:
                    IndentScope.Writer.IndWrite($"gather_paged_attention_kv_cache({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.CPU.PagedAttention pagedAttention:
                    IndentScope.Writer.IndWrite($"paged_attention<{pagedAttention.Layout.ToC()}>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {pagedAttention.LayerId}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
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
            WriteIndWithProfiler($"{deviceFunc.Name}({string.Join(",", arguments.Select(arg => arg.Name))});\n");
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

    private CSymbol VisitBuffer(Expr buffer, bool local)
    {
        var symbol = Visit(buffer);
        if (local && ((buffer.CheckedType is DistributedType) || (buffer is TIR.Buffer b && b.DistributedType != null)))
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
            var dimensions = buffer.DistributedType is null ? buffer.Dimensions : buffer.DistributedType.TensorType.Shape.Dimensions;
            var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x is TensorConst);
            var isFixedStrides = buffer.Strides.AsValueEnumerable().All(x => x is TensorConst);
            var spanStr = Visit(buffer.MemSpan).Name;

            if (isFixedDimensions && isFixedStrides)
            {
                IndentScope.Writer.IndWrite($"({spanStr})");
            }
            else
            {
                var dimensionSymbols = dimensions.AsValueEnumerable().Select(Visit).ToArray();
                var strideSymbols = buffer.Strides.AsValueEnumerable().Select(Visit).ToArray();

                var dimensionStr = isFixedDimensions ? "{}" : KernelUtility.DimensionsToC(false, dimensionSymbols, false);
                var strideStr = isFixedStrides ? "{}" : KernelUtility.StridesToC(false, strideSymbols, false);
                IndentScope.Writer.IndWrite($"({spanStr}, {dimensionStr}, {strideStr})");
            }
        }

        IndentScope.Writer.Write($";\n");
    }
}
