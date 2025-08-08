﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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

/// <summary>
/// convert single prim function to c source.
/// </summary>
internal sealed class KernelCSourceConvertVisitor : CSourceConvertVisitor, IDisposable
{
    private readonly HashSet<string> _excludedVars = new() { "data" };
    private readonly StringBuilder _kernelBuilder;
    private readonly HashSet<TIR.PrimFunction> _refFuncs;
    private readonly HashSet<TIR.Buffer> _declaredBuffers = new(ReferenceEqualityComparer.Instance);
    private Var[]? _tensorParams;

    public KernelCSourceConvertVisitor(NTTTargetOptions targetOptions)
    {
        _kernelBuilder = new StringBuilder();
        _refFuncs = new(ReferenceEqualityComparer.Instance);
        CollectivePoolSize = 0;
        TargetOptions = targetOptions;
    }

    public NTTTargetOptions TargetOptions { get; }

    public ulong CollectivePoolSize { get; private set; }

    private Var[] TensorParams => _tensorParams ??= VisitEntry.Parameters.ToArray().OfType<Var>().Where(x => !_excludedVars.Contains(x.Name)).ToArray();

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
        var paramsExcluded = VisitEntry.Parameters.ToArray().OfType<IVar>().Where(x => !_excludedVars.Contains(x.Name)).ToArray();
        var templateHeader = TensorParams.Length == 0 ? string.Empty : $"template<{string.Join(", ", Enumerable.Range(0, TensorParams.Length).Select(x => $"class T{x}"))}>" + Environment.NewLine;
        var ctype = templateHeader +
            $"void {VisitEntry.Name}({string.Concat(paramsExcluded.Select(Visit).Select(s => $"{s.Type} {s.Name}, ").ToArray())}const std::byte *rdata, const std::byte *thread_local_rdata, const std::byte *block_local_rdata, std::byte *data, std::byte *output, nncase::ntt::runtime::thread_inout_desc *const output_descs)";
        return new(
            Declare: ctype + ";\n",
            Kernel: CSourceBuiltn.MakeKernel(ctype, _kernelBuilder.ToString()),
            CollectivePoolSize: CollectivePoolSize);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
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

            WriteDimVars();

            // 3. Function body
            using (_ = new IndentScope())
            {
                Visit(expr.Body);
            }

            // 4. Function closing
            IndentScope.Writer.IndWrite("}\n");
        }

        symbol = new(ctype, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPhysicalBuffer(PhysicalBuffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var start = Visit(expr.Start);
        string loc = (expr.Location, expr.Hierarchy) switch
        {
            (MemoryLocation.Rdata, 0) => "rdata",
            (MemoryLocation.ThreadLocalRdata, 0) => "thread_local_rdata",
            (MemoryLocation.BlockLocalRdata, 0) => "block_local_rdata",
            (MemoryLocation.Data, 0) => "data",
            (MemoryLocation.Data, 1) => "data",
            (MemoryLocation.Output, 0) => "output",
            _ => throw new NotSupportedException($"{expr.Location}, {expr.Hierarchy}"),
        };

        var ptypeName = "std::byte";
        if (expr.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata or MemoryLocation.BlockLocalRdata)
        {
            // Rdata, ThreadLocalRdata and BlockLocalRdata are const
            ptypeName = $"const {ptypeName}";
        }

        string name;
        if (expr.Size is DimConst)
        {
            var spanSize = (ulong)expr.Size.FixedValue;
            name = $"std::span<{ptypeName}, {spanSize}>({loc} + {start.Name}UL, {spanSize})";
        }
        else
        {
            var spanSize = Visit(expr.Size).Name;
            name = $"std::span<{ptypeName}>({loc} + {start.Name}UL, {spanSize})";
        }

        symbol = new(start.Type, name);
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

        var buffer = Visit(expr.Buffer);
        var start = Visit(expr.Start);

        string name;
        if (expr.Start is DimConst && expr.Size is DimConst)
        {
            var spanSize = (ulong)expr.Size.FixedValue;
            name = $"{buffer.Name}.subspan<{start.Name}, {spanSize}>()";
        }
        else
        {
            var spanSize = Visit(expr.Size).Name;
            name = $"{buffer.Name}.subspan({start.Name}, {spanSize})";
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
        var dimensionTypes = dimensions.AsValueEnumerable().Select(x => Visit(x).Type).ToArray();
        var strideTypes = expr.Strides.AsValueEnumerable().Select(x => Visit(x).Type).ToArray();
        var dtypeStr = expr.ElemType.ToC();
        var dimensionStr = $"shape_t<{StringUtility.Join(", ", dimensionTypes)}>";
        var strideStr = $"strides_t<{StringUtility.Join(", ", strideTypes)}>";

        var type = expr.MemSpan.Buffer.Location is MemoryLocation.Rdata or MemoryLocation.ThreadLocalRdata or MemoryLocation.BlockLocalRdata || expr.MemSpan.Buffer.Start is TensorConst
            ? (expr.DistributedType == null
             ? $"tensor_view<{dtypeStr}, {dimensionStr}, {strideStr}> "
             : $"sharded_tensor_view<{dtypeStr}, {dimensionStr}, {KernelUtility.ShardingToC(expr.DistributedType)}, {strideStr}> ")
            : $"tensor<{dtypeStr}, {dimensionStr}, {strideStr}> ";

        symbol = new(type, expr.Name);
        DeclBuffer(expr, symbol);
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
                Visit(item);
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
                        CollectivePoolSize = Math.Max(CollectivePoolSize, (ulong)maxSize);
                        var indices = args[0].CheckedShape.Select(e => Visit(e).Name).ToSlicing(load.NdSbp, load.Placement)[0];
                        WriteWithProfiler($"tac::tensor_boxing_load_sync<fixed_shape_t<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
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
                        CollectivePoolSize = Math.Max(CollectivePoolSize, (ulong)maxSize);
                        var indices = args[0].CheckedShape.Select(e => Visit(e).Name).ToSlicing(store.NdSbp, store.Placement)[0];
                        WriteWithProfiler($"tac::tensor_boxing_store_sync<fixed_shape_t<{string.Join(',', fullShape)}>>({indices}, {VisitBuffer(args[0], local: true).Name});\n");
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
                    WriteIndWithProfiler($"im2col({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape_v<{string.Join(",", im2col.Kernel)}>, fixed_shape_v<{string.Join(",", im2col.Stride)}>, fixed_paddings_v<{string.Join(",", im2col.Padding)}>, fixed_shape_v<{string.Join(",", im2col.VectorizedAxes)}>, fixed_shape_v<{string.Join(",", im2col.PadedNums)}>);\n");
                    break;
                case TIR.NTT.Pack vectorize:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Pack>(vectorize)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;

                case TIR.NTT.Unpack devectorize:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unpack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Unpack>(devectorize)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                        }).Result);
                    }

                    break;
                case TIR.NTT.VectorizedLayerNorm vectorizedLayerNorm:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/VectorizedLayerNorm.cshtml", new TypedKernelTemplateModel<TIR.NTT.VectorizedLayerNorm>(vectorizedLayerNorm)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).Concat(vectorizedLayerNorm.PadedNums.Select(Visit).Select(x => new KernelArgument { Symbol = x })).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.InstanceNorm instanceNorm:
                    WriteWithProfiler($"instance_norm({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {args[0].CheckedDataType.ToC()} {{ {instanceNorm.Epsilon} }}, fixed_shape_v<{string.Join(",", instanceNorm.VectorizedAxes)}>{{}}, fixed_shape_v<{string.Join(",", instanceNorm.PadedNums)}>{{}} );\n");
                    break;
                case TIR.NTT.ResizeImage resize:
                    WriteIndWithProfiler($"resize({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape_v<{string.Join(",", resize.VectorizedAxes)}>, fixed_shape_v<{string.Join(",", resize.PadedNums)}>, fixed_shape_v<{string.Join(",", resize.NewSize)}>, image_resize_mode_t::{resize.ResizeMode.ToC()}, image_resize_transformation_mode_t::{resize.TransformationMode.ToC()}, image_resize_nearest_mode_t::{resize.NearestMode.ToC()});\n");
                    break;
                case TIR.NTT.VectorizedSoftmax vectorizedsoftmax:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/VectorizedSoftmax.cshtml", new TypedKernelTemplateModel<TIR.NTT.VectorizedSoftmax>(vectorizedsoftmax)
                        {
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            Args = args.ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.VectorizedBinary vectorizedBinary:
                    {
                        WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                        {
                            BinaryOp = vectorizedBinary.BinaryOp,
                            Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                        }).Result);
                    }

                    break;
                case TIR.NTT.Conv2D conv:
                    WriteIndWithProfiler($"conv2d({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, fixed_shape_v<{string.Join(", ", conv.Stride)}>, fixed_paddings_v<{string.Join(", ", conv.Padding)}>, fixed_shape_v<{string.Join(",", conv.Dilation)}>, {conv.Groups}_dim);\n");
                    break;
                case TIR.NTT.Matmul matmul:
                    {
                        var dimInfo = IR.NTT.VectorizedMatMul.GetDimInfo(matmul.TransposeA, matmul.TransposeB, args[0].CheckedShape.Rank, args[1].CheckedShape.Rank);
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.NTT.Matmul>(matmul)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "matmul");
                        if (args[0] is TIR.Buffer a && a.DistributedType?.AxisPolicies[dimInfo.Lk] is SBPSplit s)
                        {
                            var reduceKind = "tar::reduce_kind::" + string.Join("_", Enumerable.Range(0, TargetOptions.HierarchyNames.Length).Select(i => (s.Axes.Contains(i) ? "r" : string.Empty) + TargetOptions.HierarchyNames[i]));
                            WriteIndWithProfiler($"tac::tensor_reduce_sync<reduce_op::{ReduceOp.Sum.ToC()}, {reduceKind}>({VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                        }
                    }

                    break;
                case TIR.NTT.SUMMA summa:
                    var rdKind = "tar::reduce_kind::" + string.Join("_", Enumerable.Range(0, TargetOptions.HierarchyNames.Length).Select(i => i >= TargetOptions.HierarchyNames.Length - 2 ? "r" + TargetOptions.HierarchyNames[i] : string.Empty + TargetOptions.HierarchyNames[i]));
                    IndentScope.Writer.IndWrite($"{{tac::detail::tensor_reduce_sync_impl<reduce_op::sum, {rdKind}> impl; impl.reduce_group_sync();\n");
                    IndentScope.Writer.IndWrite($"summa<false>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name}, {VisitBuffer(args[2], local: false).Name}, fixed_shape_v<{string.Join(",", summa.LhsVectorizedAxes)}>, fixed_shape_v<>, fixed_shape_v<{string.Join(",", summa.RhsVectorizedAxes)}>, fixed_shape_v<>);\n");
                    IndentScope.Writer.IndWrite($"impl.reduce_group_sync();}}\n");
                    break;
                case TIR.NTT.PackedMatMul matmul:
                    {
                        var dimInfo = IR.NTT.VectorizedMatMul.GetDimInfo(false, true, args[0].CheckedShape.Rank, args[1].CheckedShape.Rank);
                        WriteWithProfiler(
                            RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedMatMul.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedMatMul>(matmul)
                            {
                                Arguments = args.Select(x => new KernelArgument { Symbol = VisitBuffer(x, local: true) }).ToArray(),
                            }).Result,
                            "packed_matmul");
                        if (args[0] is TIR.Buffer a && a.DistributedType?.AxisPolicies[dimInfo.Lk] is SBPSplit s)
                        {
                            var reduceKind = "tar::reduce_kind::" + string.Join("_", Enumerable.Range(0, TargetOptions.HierarchyNames.Length).Select(i => (s.Axes.Contains(i) ? "r" : string.Empty) + TargetOptions.HierarchyNames[i]));
                            WriteIndWithProfiler($"tac::tensor_reduce_sync<reduce_op::{ReduceOp.Sum.ToC()}, {reduceKind}>({VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                        }
                    }

                    break;
                case TIR.Memcopy copy:
                    WriteWithProfiler($"tensor_copy({VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[0], local: true).Name});\n");
                    break;
                case TIR.NTT.Gather gather:
                    {
                        WriteWithProfiler($"gather({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {gather.Axis}_dim);\n");
                        if (args[0] is TIR.Buffer b && b.DistributedType?.AxisPolicies[gather.Axis] is SBPSplit s)
                        {
                            var reduceKind = "tar::reduce_kind::" + string.Join("_", Enumerable.Range(0, TargetOptions.HierarchyNames.Length).Select(i => (s.Axes.Contains(i) ? "r" : string.Empty) + TargetOptions.HierarchyNames[i]));
                            WriteIndWithProfiler($"tac::tensor_reduce_sync<reduce_op::{ReduceOp.Sum.ToC()}, {reduceKind}>({VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                        }
                    }

                    break;
                case TIR.NTT.Swish swish:
                    if (swish.Beta != 1.0f)
                    {
                        IndentScope.Writer.IndWrite($"\n{{\nauto b= {swish.Beta}; auto tb = make_tensor_view_from_address<float>(&b, fixed_shape_v<>);\n");
                        WriteIndWithProfiler($"binary<ops::swishb>({VisitBuffer(args[0], local: true).Name}, tb, {VisitBuffer(args[1], local: true).Name});\n}}\n");
                    }
                    else
                    {
                        WriteWithProfiler($"unary<ops::swish>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    }

                    break;
                case TIR.NTT.Slice slice:
                    WriteWithProfiler($"slice({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {VisitDimOrShape(args[1]).Name}, {VisitDimOrShape(args[2]).Name}, fixed_dims_v<{string.Join(",", slice.Axes)}>, fixed_dims_v<{string.Join(",", slice.Strides)}>);\n");
                    break;
                case TIR.NTT.Concat concat:
                    WriteWithProfiler($"concat(std::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name}, {concat.Axis}_dim);\n");
                    break;
                case TIR.NTT.Transpose transpose:
                    WriteWithProfiler($"transpose({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_dims_v<{string.Join(",", transpose.Perm)}>);\n");
                    break;
                case TIR.NTT.Pad pad:
                    var padValueType = args[0].CheckedTensorType.DType is VectorType vt ? vt.ElemType : args[0].CheckedTensorType.DType;
                    WriteWithProfiler($"pad({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitDimOrShape(args[1]).Name}, {args[0].CheckedDataType.ToC()} {{ ({padValueType.ToC()}){pad.PadValue} }});\n");
                    break;
                case TIR.NTT.Reduce reduce:
                    WriteWithProfiler($"reduce_{reduce.ReduceOp.ToC()}({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape_v<{string.Join(",", reduce.Axes)}>, fixed_shape_v<{string.Join(",", reduce.VectorizedAxes)}>);\n");
                    break;
                case TIR.NTT.ReduceArg reduceArg:
                    WriteWithProfiler($"reduce_arg<ops::{reduceArg.ReduceArgOp.ToC()[4..]}, {reduceArg.Axis}, {reduceArg.SelectLastIndex.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}, {reduceArg.KeepDims.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Clamp clamp:
                    string min = clamp.Min is float.NegativeInfinity ? float.MinValue.ToString() : clamp.Min.ToString();
                    string max = clamp.Max is float.PositiveInfinity ? float.MaxValue.ToString() : clamp.Max.ToString();
                    WriteWithProfiler($"clamp({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, (float){min}, (float){max});\n");
                    break;
                case TIR.NTT.Cast cast:
                    WriteWithProfiler($"cast({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, fixed_shape_v<{string.Join(",", cast.VectorizeAxes.ToArray())}>);\n");
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
                        if (grs.InType.AxisPolicies.Any(s => s is SBPPartial))
                        {
                            // deprecated
                            var sbpPartial = (SBPPartial)grs.InType.AxisPolicies.Where(s => s is SBPPartial).Distinct().First();
                            var reduceKind = "tar::reduce_kind::" + string.Join("_", grs.InType.AxisPolicies.Select((s, i) => (s is SBPPartial ? "r" : string.Empty) + TargetOptions.HierarchyNames[i]));
                            WriteIndWithProfiler($"tac::tensor_reduce_sync<reduce_op::{sbpPartial.Op.ToC()}, {reduceKind}>({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                        }
                        else
                        {
                            (var maxSize, _) = TensorUtilities.GetTensorMaxSizeAndStrides(args[0].CheckedTensorType);
                            CollectivePoolSize = Math.Max(CollectivePoolSize, (ulong)maxSize);
                            WriteWithProfiler($"reshard({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: false).Name});\n");
                        }
                    }

                    break;
                case TIR.NTT.GetItem getItem:
                    IndentScope.Writer.Write($"get_item({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitDimOrShape(args[1]).Name});\n");
                    break;
                case TIR.NTT.Stack stack:
                    IndentScope.Writer.Write($"stack<{stack.Axis}>(std::make_tuple({string.Join(",", args.SkipLast(1).Select(x => VisitBuffer(x, local: true)).Select(s => s.Name))}), {VisitBuffer(args[^1], local: true).Name});\n");
                    break;
                case TIR.NTT.Reshape reshape:
                    IndentScope.Writer.Write($"reshape({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.ShapeOf shapeOf:
                    IndentScope.Writer.Write($"shapeof({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name});\n");
                    break;
                case TIR.NTT.Range range:
                    IndentScope.Writer.Write($"range({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name},{VisitBuffer(args[2], local: true).Name},{VisitBuffer(args[3], local: true).Name});\n");
                    break;
                case TIR.NTT.ConstantOfShape constantOfShape:
                    IndentScope.Writer.Write($"constant_of_shape({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.NTT.UpdatePagedAttentionKVCache updatePagedAttentionKVCache:
                    IndentScope.Writer.IndWrite($"update_paged_attention_kv_cache<caching::attention_cache_kind::{updatePagedAttentionKVCache.CacheKind.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture)}>({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {updatePagedAttentionKVCache.LayerId}, {updatePagedAttentionKVCache.Layout.ToC()});\n");
                    break;
                case TIR.NTT.GatherPagedAttentionKVCache gakv:
                    IndentScope.Writer.IndWrite($"gather_paged_attention_kv_cache({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name});\n");
                    break;
                case TIR.NTT.PagedAttention pagedAttention:
                    IndentScope.Writer.IndWrite($"paged_attention({VisitBuffer(args[0], local: false).Name}, {VisitBuffer(args[1], local: true).Name}, {VisitBuffer(args[2], local: true).Name}, {VisitBuffer(args[3], local: true).Name}, {pagedAttention.LayerId}, {VisitBuffer(args[4], local: true).Name}, {pagedAttention.Layout.ToC()});\n");
                    break;
                case TIR.NTT.GetPositionIds getPositionIds:
                    IndentScope.Writer.IndWrite($"get_position_ids({VisitBuffer(args[0], local: true).Name}, {VisitBuffer(args[1], local: false).Name});\n");
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
                Visit(item);
            }
#if DEBUG_PRINT
            IndentScope.Writer.IndWrite($"runtime_util->printf(\"call {deviceFunc.Name} bid %d tid %d\\n\", bid, tid);\n");
#endif
            var argumentNames = new List<string>();
            string? dataName = null;
            foreach (var arg in expr.Arguments)
            {
                if (arg is TIR.Buffer b)
                {
                    var buffer = VisitBuffer(b, local: true);
                    if (b.Name.StartsWith("data_"))
                    {
                        dataName = $"rdata, thread_local_rdata, block_local_rdata, (std::byte *){buffer.Name}.buffer().data(), output, output_descs";
                    }
                    else
                    {
                        argumentNames.Add(buffer.Name);
                    }
                }
                else
                {
                    argumentNames.Add(Visit(arg).Name);
                }
            }

            if (dataName is not null)
            {
                argumentNames.Add(dataName);
            }

            _refFuncs.Add(deviceFunc);
            WriteIndWithProfiler($"{deviceFunc.Name}({string.Join(", ", argumentNames)});\n");
        }
        else
        {
            var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
            switch (expr.Target)
            {
                case IR.Math.Binary op:
                    str = CSourceUtilities.ConvertBinary(op, arguments);
                    break;
                case IR.Math.Unary op:
                    str = CSourceUtilities.ConvertUnary(op, arguments);
                    break;
                case IR.Math.Compare op:
                    str = CSourceUtilities.ConvertCompare(op, arguments);
                    break;
                case IR.Math.Select op:
                    str = CSourceUtilities.ConvertSelect(op, arguments);
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
                case IR.Math.Clamp op:
                    str = CSourceUtilities.ConvertClamp(op, arguments);
                    break;
                case IR.Shapes.AsTensor op:
                    str = CSourceUtilities.ConvertAsTensor(op, arguments);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported call target: {expr.Target}");
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
                var name = Visit(call).Name;
                if (call.Target is not IR.Shapes.AsTensor)
                {
                    IndentScope.Writer.IndWrite(name);
                }
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
            IndentScope.Writer.IndWrite($"{value.Name}.shape().copy_to(output_descs[{i}].shape);\n");
            IndentScope.Writer.IndWrite($"{value.Name}.strides().copy_to(output_descs[{i}].strides);\n");
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

    private void DeclBuffer(TIR.Buffer buffer, CSymbol symbol)
    {
        if (_declaredBuffers.Add(buffer))
        {
            IndentScope.Writer.IndWrite($"auto {symbol.Name}");
            if (buffer.MemSpan.Buffer.Start is not None)
            {
                // If the buffer has a start, we create a tensor view
                var dtypeStr = buffer.ElemType.ToC();
                var dimensions = buffer.DistributedType is null ? buffer.Dimensions : ((RankedShape)buffer.DistributedType.TensorType.Shape).Dimensions;
                var spanStr = $"span_cast<{dtypeStr}>({Visit(buffer.MemSpan).Name})";
                var dimensionValues = dimensions.AsValueEnumerable().Select(x => Visit(x).Name);
                var strideValues = buffer.Strides.AsValueEnumerable().Select(x => Visit(x).Name);

                if (buffer.DistributedType is DistributedType distributedType)
                {
                    IndentScope.Writer.IndWrite($"= make_sharded_tensor_view({spanStr}, make_shape({StringUtility.Join(", ", dimensionValues)}), {KernelUtility.ShardingToC(distributedType)}, make_strides({StringUtility.Join(", ", strideValues)}))");
                }
                else
                {
                    IndentScope.Writer.IndWrite($"= make_tensor_view({spanStr}, make_shape({StringUtility.Join(", ", dimensionValues)}), make_strides({StringUtility.Join(", ", strideValues)}))");
                }
            }

            IndentScope.Writer.Write($";\n");
        }
    }
}
