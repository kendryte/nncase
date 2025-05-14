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
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

/// <summary>
/// convert single prim function to c source.
/// </summary>
internal sealed class KernelCSourceConvertVisitor : CSourceConvertVisitor, IDisposable
{
    private readonly StringBuilder _kernelBuilder;

    private readonly StringBuilder _sharedBuilder;
    private readonly HashSet<TIR.PrimFunction> _refFuncs;
    private readonly StringWriter _sharedWriter;
    private Var[]? _tensorParams;
    private ulong _collective_pool_size;

    public KernelCSourceConvertVisitor(ulong dataAlign, ulong dataUsage, ulong rdataPoolSize, ulong localRdataPoolSize, NTTTargetOptions targetOptions)
    {
        DataAlign = dataAlign;
        DataUsage = dataUsage;
        RdataPoolSize = rdataPoolSize;
        LocalRdataPoolSize = localRdataPoolSize;
        _kernelBuilder = new StringBuilder();
        _sharedBuilder = new StringBuilder();
        _sharedWriter = new StringWriter(_sharedBuilder);
        _refFuncs = new(ReferenceEqualityComparer.Instance);
        _collective_pool_size = 0;
        TargetOptions = targetOptions;
    }

    public PrimFunction VisitEntry => (TIR.PrimFunction)VisitRoot!;

    public int CallCount { get; private set; }

    public NTTTargetOptions TargetOptions { get; }

    public ulong DataAlign { get; }

    public ulong DataUsage { get; }

    public ulong RdataPoolSize { get; }

    public ulong LocalRdataPoolSize { get; }

    private Var[] TensorParams => _tensorParams ??= VisitEntry.Parameters.ToArray().OfType<Var>().ToArray();

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
        var templateHeader = TensorParams.Length == 0 ? string.Empty : $"template<{string.Join(", ", Enumerable.Range(0, TensorParams.Length).Select(x => $"class T{x}"))}>" + Environment.NewLine;
        var ctype = templateHeader +
            $"void {VisitEntry.Name}({string.Concat(VisitEntry.Parameters.AsValueEnumerable().Select(Visit).Select(s => $"{s.Type} {s.Name}, ").ToArray().Concat(_exprMemo.Keys.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata).Select(Visit).Select(s => $" {s.Type} {s.Name}, ").ToArray()))}std::byte *data, std::byte *output, nncase::ntt::runtime::thread_inout_desc *const output_descs)";
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
        var index = TensorParams.IndexOf(expr);
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

        var returnType = ((CallableType)expr.CheckedType).ReturnType;
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
            (MemoryLocation.Output, 0) => "output",
            _ => throw new NotSupportedException($"{expr.Location}, {expr.Hierarchy}"),
        };
        var ptype = (PointerType)expr.CheckedDataType;
        var ptypeName = ptype.ElemType.ToC();
        string name;
        if (expr.Size is DimConst)
        {
            var spanSize = (ulong)expr.Size.FixedValue / (ulong)ptype.ElemType.SizeInBytes;
            name = $"std::span<{ptypeName}, {spanSize}> (reinterpret_cast<{ptypeName}*>({loc} + {start.Name}UL), {spanSize})";
        }
        else
        {
            var spanSize = $"{Visit(expr.Size).Name} / {ptype.ElemType.SizeInBytes}";
            name = $"std::span<{ptypeName}> (reinterpret_cast<{ptypeName}*>({loc} + {start.Name}UL), {spanSize})";
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

        var dimensions = expr.DistributedType is null ? expr.Dimensions : ((RankedShape)expr.DistributedType.TensorType.Shape).Dimensions;
        var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x.IsFixed);
        var isFixedStrides = expr.Strides.AsValueEnumerable().All(x => x.IsFixed);
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
        if (expr.Target is Op kop && kop is TIR.NTT.NTTKernelOp or TIR.Memcopy)
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
                case TIR.NTT.Unary unary:
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        UnaryOp = unary.UnaryOp,
                    }).Result);
                    break;
                case TIR.NTT.TensorLoad load:
                    if (args.Length == 1)
                    {
                        var fullShape = args[0].CheckedShape.ToValueArray();
                        (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                        _collective_pool_size = Math.Max(_collective_pool_size, (ulong)maxSize);
                        var indices = args[0].CheckedShape.Select(e => Visit(e).Name).ToSlicing(load.NdSbp, load.Placement)[0];
                        WriteWithProfiler($"tac::tensor_boxing_load_sync<fixed_shape<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
                    }
                    else
                    {
                        WriteWithProfiler($"reshard({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: false).Name});\n");
                    }

                    break;
                case TIR.NTT.TensorStore store:
                    if (args.Length == 1)
                    {
                        var fullShape = args[0].CheckedShape.ToValueArray();
                        (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                        _collective_pool_size = Math.Max(_collective_pool_size, (ulong)maxSize);
                        var indices = args[0].CheckedShape.Select(e => Visit(e).Name).ToSlicing(store.NdSbp, store.Placement)[0];
                        WriteWithProfiler($"tac::tensor_boxing_store_sync<fixed_shape<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
                    }
                    else
                    {
                        WriteWithProfiler($"reshard({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    }

                    break;
                case TIR.NTT.Binary binary:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            BinaryOp = binary.BinaryOp,
                        }).Result);
                    }

                    break;
                case TIR.NTT.Im2col im2col:
                    WriteIndWithProfiler($"im2col({VisitBuffer(args[0], local: true).Name}, fixed_shape<{string.Join(",", im2col.Kernel)}>{{}}, fixed_shape<{string.Join(",", im2col.Stride)}>{{}}, fixed_shape<{string.Join(",", im2col.Padding)}>{{}}, fixed_shape<{string.Join(",", im2col.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", im2col.PadedNums)}>{{}}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Pack pack:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Pack>(pack)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;

                case TIR.NTT.Unpack unpack:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unpack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Unpack>(unpack)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;
                case TIR.NTT.PackedLayerNorm packedLayerNorm:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedLayerNorm.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedLayerNorm>(packedLayerNorm)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.InstanceNorm instanceNorm:
                    WriteWithProfiler($"instance_norm({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {args[0].CheckedDataType.ToC()} {{ {instanceNorm.Epsilon} }}, fixed_shape<{string.Join(",", instanceNorm.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", instanceNorm.PadedNums)}>{{}} );\n");
                    break;
                case TIR.NTT.ResizeImage resize:
                    WriteIndWithProfiler($"resize({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape<{string.Join(",", resize.PackedAxes)}>{{}}, fixed_shape<{string.Join(",", resize.PadedNums)}>{{}}, fixed_shape<{string.Join(",", resize.NewSize)}>{{}}, image_resize_mode_t::{resize.ResizeMode.ToC()}, image_resize_transformation_mode_t::{resize.TransformationMode.ToC()}, image_resize_nearest_mode_t::{resize.NearestMode.ToC()});\n");
                    break;
                case TIR.NTT.PackedSoftmax packedsoftmax:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedSoftMax.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedSoftmax>(packedsoftmax)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.PackedBinary packedBinary:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                        {
                            BinaryOp = packedBinary.BinaryOp,
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.Conv2D conv:
                    WriteIndWithProfiler($"conv2d({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, fixed_shape<{string.Join(",", conv.Stride)}>{{}}, fixed_shape<{string.Join(",", conv.Padding)}>{{}}, fixed_shape<{string.Join(",", conv.Dilation)}>{{}}, {conv.Groups});\n");
                    break;
                case TIR.NTT.Matmul matmul:
                    IndentScope.Writer.IndWrite(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.NTT.Matmul>(matmul)
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                    }).Result);

                    break;
                case TIR.NTT.SUMMA summa:
                    var rdKind = "tar::reduce_kind::" + string.Join("_", Enumerable.Range(0, TargetOptions.HierarchyNames.Length).Select(i => i >= TargetOptions.HierarchyNames.Length - 2 ? "r" + TargetOptions.HierarchyNames[i] : string.Empty + TargetOptions.HierarchyNames[i]));
                    IndentScope.Writer.IndWrite($"{{tac::detail::tensor_reduce_sync_impl<reduce_op::sum, {rdKind}> impl; impl.reduce_group_sync();\n");
                    IndentScope.Writer.IndWrite($"summa<false>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name}, {VisitBuffer(args[2], local: false).Name}, fixed_shape<{string.Join(",", summa.LhsPackedAxes)}>{{}}, fixed_shape<{string.Join(",", summa.LhsPadedNums)}>{{}}, fixed_shape<{string.Join(",", summa.RhsPackedAxes)}>{{}}, fixed_shape<{string.Join(",", summa.RhsPadedNums)}>{{}});\n");
                    IndentScope.Writer.IndWrite($"impl.reduce_group_sync();}}\n");
                    break;
                case TIR.Memcopy copy:
                    WriteWithProfiler($"tensor_copy({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: true).Name});\n");
                    break;
                case TIR.NTT.Gather gather:
                    WriteWithProfiler($"gather<{gather.Axis}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.NTT.Swish swish:
                    if (swish.Beta != 1.0f)
                    {
                        throw new NotSupportedException();
                    }

                    WriteWithProfiler($"unary<ops::swish>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Slice slice:
                    WriteWithProfiler($"slice<fixed_dims<int64_t, {string.Join(",", slice.Axes)}>, fixed_dims<int64_t, {string.Join(",", slice.Strides)}>>({VisitBuffer(args[0], local: true).Name}, {VisitDimOrShape(args[1]).Name}, {VisitDimOrShape(args[2]).Name}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.NTT.Concat concat:
                    WriteWithProfiler($"concat<{concat.Axis}>(std::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name});\n");
                    break;
                case TIR.NTT.Transpose transpose:
                    WriteWithProfiler($"transpose<fixed_shape<{string.Join(",", transpose.Perm)}>>({VisitBuffer(args[0], local: true).Name}, {VisitDimOrShape(args[1]).Name});\n");
                    break;
                case TIR.NTT.Pad pad:
                    WriteWithProfiler($"pad<{string.Join(",", pad.Paddings)}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {args[0].CheckedDataType.ToC()} {{ {pad.PadValue} }} );\n");
                    break;
                case TIR.NTT.Reduce reduce:
                    WriteWithProfiler($"reduce_{reduce.ReduceOp.ToC()}<fixed_shape<{string.Join(",", reduce.Axes)}>, fixed_shape<{string.Join(",", reduce.PackedAxes)}>, fixed_shape<{string.Join(",", reduce.PadedNums)}>>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.ReduceArg reduceArg:
                    WriteWithProfiler($"reduce_arg<ops::{reduceArg.ReduceArgOp.ToC()[4..]}, {reduceArg.Axis}, {reduceArg.SelectLastIndex.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, {reduceArg.KeepDims.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape<>{{}}, fixed_shape<>{{}});\n");
                    break;
                case TIR.NTT.Clamp clamp:
                    string min = clamp.Min is float.NegativeInfinity ? float.MinValue.ToString() : clamp.Min.ToString();
                    string max = clamp.Max is float.PositiveInfinity ? float.MaxValue.ToString() : clamp.Max.ToString();
                    WriteWithProfiler($"clamp({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, (float){min}, (float){max});\n");
                    break;
                case TIR.NTT.Cast cast:
                    WriteWithProfiler($"cast({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Where where:
                    WriteWithProfiler($"where({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.NTT.Expand expand:
                    WriteWithProfiler($"expand({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Erf erf:
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                    {
                        Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        UnaryOp = UnaryOp.Erf,
                    }).Result);
                    break;
                case TIR.NTT.Compare compare:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Compare.cshtml", new CompareKernelTemplateModel
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            CompareOp = compare.CompareOp,
                        }).Result);
                    }

                    break;
                case TIR.NTT.ScatterND scatterND:
                    WriteWithProfiler($"scatter_nd({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name});\n");

                    break;
                case TIR.NTT.GatherReduceScatter grs:
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
                case TIR.NTT.GetItem getItem:
                    IndentScope.Writer.Write($"get_item({VisitBuffer(args[0], local: true).Name}, {VisitDimOrShape(args[1]).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.NTT.Stack stack:
                    IndentScope.Writer.Write($"stack<{stack.Axis}>(std::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name});\n");
                    break;
                case TIR.NTT.Reshape reshape:
                    IndentScope.Writer.Write($"reshape({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
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
                case TIR.NTT.PtrOf op:
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

    protected override CSymbol VisitIfThenElse(IfThenElse expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var condition = Visit(expr.Condition);
        IndentScope.Writer.IndWrite($"if ({condition.Name}) {{\n");
        using (new IndentScope())
        {
            Visit(expr.Then);
        }

        if (expr.Else.Count > 0)
        {
            IndentScope.Writer.IndWrite("} else {\n");
            using (new IndentScope())
            {
                Visit(expr.Else);
            }
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

        var var = Visit(expr.Var);
        var value = Visit(expr.Expression);
        IndentScope.Writer.IndWrite($"{var.Type} {var.Name} = {value.Name};\n");
        using (new IndentScope())
        {
            Visit(expr.Body);
        }

        var body = Visit(expr.Body);
        symbol = new(body.Type, body.Name);
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

    protected override CSymbol VisitReturn(Return expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var values = expr.Values.AsValueEnumerable().Select(Visit).ToArray();
        for (int i = 0; i < values.Length; i++)
        {
            var value = values[i];
            var rank = expr.Values[i].CheckedShape.Rank;
            IndentScope.Writer.IndWrite($"output_descs[{i}].data = (std::byte *){value.Name}.elements().data();\n");
            IndentScope.Writer.IndWrite($"output_descs[{i}].size = {value.Name}.size() * sizeof({expr.Values[i].CheckedDataType.ToC()});\n");
            IndentScope.Writer.IndWrite($"std::copy_n({value.Name}.shape().begin(), {rank}, output_descs[{i}].shape);\n");
            IndentScope.Writer.IndWrite($"std::copy_n({value.Name}.strides().begin(), {rank}, output_descs[{i}].strides);\n");
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    private CSymbol VisitBuffer(BaseExpr buffer, bool local)
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
            var dimensions = buffer.DistributedType is null ? buffer.Dimensions : ((RankedShape)buffer.DistributedType.TensorType.Shape).Dimensions;
            var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x.IsFixed);
            var isFixedStrides = buffer.Strides.AsValueEnumerable().All(x => x.IsFixed);
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
