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
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

public class DeviceCSourceConvertVisitor : CSourceConvertVisitor
{
    protected readonly StringBuilder _deviceBuilder;

    public DeviceCSourceConvertVisitor()
    {
        _deviceBuilder = new();
    }

    public static void WriteWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal);
            if (index != -1)
            {
                tagName = functionName.Substring(0, index);
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
#if false // Disable device profiling for now.
        IndentScope.Writer.Write($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.Write($"auto_profiler profiler(function_name, runtime::profiling_level::device);\n");
#endif
        IndentScope.Writer.Write($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public static void WriteIndWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal);
            if (index != -1)
            {
                tagName = functionName.Substring(0, index);
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
#if false // Disable device profiling for now.
        IndentScope.Writer.IndWrite($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.IndWrite($"auto_profiler profiler(function_name, runtime::profiling_level::device);\n");
#endif
        IndentScope.Writer.IndWrite($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

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

        var ctype = $"template<{string.Join(", ", Enumerable.Range(0, expr.Parameters.Length).Select(x => $"class T{x}"))}>" + Environment.NewLine +
            $"void {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit).Select((s, i) => $"T{i} &&{s.Name}").ToArray())})";

        using (var scope = new IndentScope(_deviceBuilder))
        {
            // 1. Function signature
            IndentScope.Writer.IndWrite($"{ctype} {{\n");

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
        _exprMemo[(BaseExpr)expr.Var] = new(value.Type, @var.Name);

#if DEBUG_PRINT
        IndentScope.Writer.IndWrite($"runtime_util->printf(\"let {@var.Name}\\n\");\n");
#endif
        if (value.Type.StartsWith("array"))
        {
            var ss = value.Type.Split(" ");
            IndentScope.Writer.IndWrite($"{ss[1]} {@var.Name}[{ss[2]}];\n");
        }
        else
        {
            IndentScope.Writer.IndWrite($"{value.Type} {@var.Name} = {value.Name};\n");
        }

        Visit(expr.Body);

        symbol = new(string.Empty, string.Empty);
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
        var size = Visit(expr.Size);
        string name = expr.Location switch
        {
            MemoryLocation.L2Data => $"L2Data + {start.Name}",
            MemoryLocation.L1Data => $"L1Data + {start.Name}",
            MemoryLocation.Input or MemoryLocation.Output => start.Name,
            _ => throw new NotSupportedException(expr.Location.ToString()),
        };

        var str = $"std::span<std::byte, {size.Name}>({name} + {start.Name}, {size.Name})";
        symbol = new(start.Type, str);
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
        var size = Visit(expr.Size);

        var str = $"{buffer.Name}.subspan<{start.Name}, {size.Name}>()";
        symbol = new(start.Type, str);
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
        var type = $"tensor_view<{dtypeStr}, {dimensionStr}, {strideStr}> ";

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
            TensorType or DistributedType => "auto",
            _ => throw new NotSupportedException(),
        };

        string str = string.Empty;
        var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        switch (expr.Target)
        {
            case PrimFunction deviceFunc:
                WriteIndWithProfiler($"{deviceFunc.Name}({string.Join(",", arguments.Select(arg => arg.Name))});\n");
                break;
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
            case IR.Shapes.AsTensor op:
                str = arguments[0].Name;
                break;
            case TIR.NTT.SramPtr op:
                str = $"g_cpu_mt->sram_address(bid, tid) + {arguments[0].Name}";
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
            case TIR.NTT.PtrOf op:
                str = op.PtrName + ".data()";
                break;
            case IR.Buffers.Allocate op:
                if (op.Malloc)
                {
                    str = $"({type})runtime_util->malloc({arguments[0].Name})";
                }
                else
                {
                    type = $"array {((PointerType)expr.CheckedDataType).ElemType.ToC()} {arguments[0].Name}";
                    str = $"";
                }

                break;
            case IR.Buffers.BufferSubview op:
                {
                    var arg0 = VisitDimOrShape(expr.Arguments[1], CShapeKind.Shape).Name;
                    var arg1 = VisitDimOrShape(expr.Arguments[2], CShapeKind.Shape).Name;
                    str = $"{arguments[0].Name}.view({arg0}, {arg1})";
                }

                break;
            case IR.Buffers.AllocateBufferView op:
                {
                    var buffer = (TIR.Buffer)expr.Arguments[0];
                    var dimensions = buffer.DistributedType is null ? buffer.Dimensions : ((RankedShape)buffer.DistributedType.TensorType.Shape).Dimensions;
                    var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x.IsFixed);
                    var isFixedStrides = buffer.Strides.AsValueEnumerable().All(x => x.IsFixed);
                    var dimensionSymbols = dimensions.AsValueEnumerable().Select(Visit).ToArray();
                    var strideSymbols = buffer.Strides.AsValueEnumerable().Select(Visit).ToArray();

                    var dtypeStr = buffer.ElemType.ToC();
                    var dimensionStr = KernelUtility.DimensionsToC(isFixedDimensions, dimensionSymbols, false);
                    var strideStr = KernelUtility.StridesToC(isFixedStrides, strideSymbols, false);
                    str = $"{{span_cast<{dtypeStr}>({Visit(buffer.MemSpan).Name}), {dimensionStr}, {strideStr}}}";
                }

                break;
            case IR.Tensors.Cast op:
                str = $"(({op.NewType.ToC()}){arguments[0].Name})";
                break;
            case TIR.Memcopy op:
                WriteIndWithProfiler($"tensor_copy({arguments[1].Name}, {arguments[0].Name});\n");
                break;
            case TIR.NTT.Unary op:
                WriteIndWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    UnaryOp = op.UnaryOp,
                }).Result);
                break;
            case TIR.NTT.Binary op:
                WriteIndWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    BinaryOp = op.BinaryOp,
                }).Result);
                break;
            case TIR.NTT.VectorizedBinary op:
                WriteIndWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    BinaryOp = op.BinaryOp,
                }).Result);
                break;
            case TIR.NTT.Swish swish:
                if (swish.Beta == 1.0f)
                {
                    WriteIndWithProfiler($"unary<ops::swish>({arguments[0].Name}, {arguments[1].Name});\n");
                }
                else
                {
                    IndentScope.Writer.IndWrite($"\n{{\nauto b= {swish.Beta}; auto tb = make_tensor_view_from_address<float>(&b, fixed_shape_v<>);\n");
                    WriteIndWithProfiler($"binary<ops::swishb>({arguments[0].Name}, tb, {arguments[1].Name});\n}}\n");
                }

                break;
            case TIR.NTT.Matmul matmul:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.NTT.Matmul>(matmul)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.PackedMatMul matmul:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedMatMul.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedMatMul>(matmul)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.Pack vectorize:
                WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Pack>(vectorize)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Transpose transpose:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Transpose.cshtml", new TypedKernelTemplateModel<TIR.NTT.Transpose>(transpose)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Unpack devectorize:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unpack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Unpack>(devectorize)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Reduce reduce:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Reduce.cshtml", new TypedKernelTemplateModel<TIR.NTT.Reduce>(reduce)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Cast cast:
                IndentScope.Writer.IndWrite($"cast({arguments[0].Name}, {arguments[1].Name});\n");
                break;
            default:
                throw new NotSupportedException($"Unsupported call target: {expr.Target}");
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

    protected override CSymbol VisitTupleConst(TupleConst tp)
    {
        if (_exprMemo.TryGetValue(tp, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = $"{string.Join(",", tp.Value.Select(x => Visit(Const.FromValue(x)).Name))}";
        symbol = new(type, str);
        _exprMemo.Add(tp, symbol);
        return symbol;
    }

    protected override CSymbol VisitTuple(IR.Tuple tp)
    {
        if (_exprMemo.TryGetValue(tp, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = $"{string.Join(",", tp.Fields.AsValueEnumerable().Select(x => Visit(x).Name).ToArray())}";
        symbol = new(type, str);
        _exprMemo.Add(tp, symbol);
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
        IndentScope.Writer.IndWrite($"for ({loopVar.Type} {loopVar.Name} = {Visit(expr.Domain.Start).Name}; {loopVar.Name} < {Visit(expr.Domain.Stop).Name}; {loopVar.Name} += {Visit(expr.Domain.Step).Name}) {{\n");
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

        var name = IRHelpers.GetIdentityName(expr.Name);
        var index = VisitEntry.Parameters.IndexOf(expr);
        if (index != -1)
        {
            symbol = new CSymbol($"T{index}", name);
        }
        else
        {
            symbol = new(
                expr.CheckedType switch
                {
                    TensorType t => t.DType.ToC(),
                    AnyType => "auto",
                    _ => throw new ArgumentOutOfRangeException(nameof(expr)),
                },
                expr.Name + "_" + expr.GlobalVarIndex.ToString());
        }

        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBufferRegion(BufferRegion expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        // FIXME: Use extents instead of stop in BufferRegion.
        throw new NotImplementedException();
#if false
        var buffer = Visit(expr.Buffer);
        var begins = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Start).Name))}";
        var extents = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Stop).Name))}";
        symbol = new(string.Empty, $"{buffer.Name}.view(make_shape({begins}), make_shape({extents}))");
        _exprMemo.Add(expr, symbol);
        return symbol;
#endif
    }
}
