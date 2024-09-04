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

namespace Nncase.CodeGen.CPU;

public class DeviceCSourceConvertVisitor : ExprFunctor<CSymbol, Unit>
{
#pragma warning disable SA1401
    protected readonly Dictionary<Expr, CSymbol> _exprMemo;
    protected readonly StringBuilder _deviceBuilder;

    public DeviceCSourceConvertVisitor()
    {
        _exprMemo = new(ReferenceEqualityComparer.Instance);
        _deviceBuilder = new();
    }

    public PrimFunction VisitEntry => (TIR.PrimFunction)VisitRoot!;

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

        var ctype = $"template<{string.Join(", ", Enumerable.Range(0, expr.Parameters.Length).Select(x => $"class T{x}"))}>" +
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
        _exprMemo[expr.Var] = new(value.Type, @var.Name);

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
    protected override CSymbol VisitMemSpan(MemSpan expr)
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

        var str = start.Type switch
        {
            "uint8_t *" => $"std::span<uint8_t, {size.Name}>({name}, {size.Name})",
            "auto" => $"std::span({name})",
            string s when s.StartsWith("array") => $"std::span({name})",
            _ => throw new NotSupportedException(start.Type),
        };

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

        var type = $"tensor_view<{expr.ElemType.ToC()}, {KernelUtility.DimensionsToC(expr.Dimensions)}, {KernelUtility.StridesToC(expr.Strides)}> ";

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
            TensorType { Shape: { IsRanked: true } } x => x.Shape.IsFixed switch
            {
                true => $"tensor_view<{x.DType.ToC()}, fixed_shape<{x.Shape.ToString()[1..^1]}>>",
                false => "auto",
            },
            _ => throw new NotSupportedException(),
        };

        string str = string.Empty;
        var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        switch (expr.Target)
        {
            case PrimFunction deviceFunc:
                IndentScope.Writer.IndWrite($"{deviceFunc.Name}({string.Join(",", arguments.Select(arg => arg.Name))});\n");
                break;
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
            case TIR.CPU.SramPtr op:
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
            case TIR.CPU.PtrOf op:
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
                    var arg0 = expr.Arguments[1] switch
                    {
                        TupleConst => $"fixed_shape<{arguments[1].Name}>{{}}",
                        IR.Tuple tc => $"make_ranked_shape({arguments[1].Name})",
                        _ => throw new ArgumentOutOfRangeException(nameof(expr)),
                    };

                    var arg1 = expr.Arguments[2] switch
                    {
                        TupleConst => $"fixed_shape<{arguments[2].Name}>{{}}",
                        IR.Tuple tc => $"make_ranked_shape({arguments[2].Name})",
                        _ => throw new ArgumentOutOfRangeException(nameof(expr)),
                    };

                    str = $"{arguments[0].Name}.view({arg0}, {arg1})";
                }

                break;
            case IR.Buffers.AllocateBufferView op:
                {
                    var buffer = (TIR.Buffer)expr.Arguments[0];
                    if (buffer.CheckedShape.IsFixed)
                    {
                        str = $"{{span_cast<{buffer.ElemType.ToC()}>({Visit(buffer.MemSpan).Name}), {KernelUtility.DimensionsToC(buffer.Dimensions)}{{}}, {KernelUtility.StridesToC(buffer.Strides)}{{}}}}";
                    }
                    else
                    {
                        str = $"{{span_cast<{buffer.ElemType.ToC()}>({Visit(buffer.MemSpan).Name}), make_ranked_shape({StringUtility.Join(", ", buffer.Dimensions.AsValueEnumerable().Select(x => Visit(x).Name))})}}";
                    }
                }

                break;
            case IR.Tensors.Cast op:
                str = $"(({op.NewType.ToC()}){arguments[0].Name})";
                break;
            case TIR.CPU.Memcopy op:
                IndentScope.Writer.IndWrite($"tensor_copy({arguments[1].Name}, {arguments[0].Name});\n");
                break;
            case TIR.CPU.Unary op:
                IndentScope.Writer.IndWrite(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    UnaryOp = op.UnaryOp,
                }).Result);
                break;
            case TIR.CPU.Binary op:
                IndentScope.Writer.IndWrite(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    BinaryOp = op.BinaryOp,
                }).Result);
                break;
            case TIR.CPU.PackedBinary op:
                IndentScope.Writer.IndWrite(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    BinaryOp = op.BinaryOp,
                }).Result);
                break;
            case TIR.CPU.Swish swish:
                if (swish.Beta == 1.0f)
                {
                    IndentScope.Writer.IndWrite($"unary<ops::swish>({arguments[0].Name}, {arguments[1].Name});\n");
                }
                else
                {
                    IndentScope.Writer.IndWrite($"float beta[1] = {{{swish.Beta}}};\n");
                    IndentScope.Writer.IndWrite($"tensor_view<float, fixed_shape<1>> tb(std::span<float, 1>(beta, beta + 1));\n");
                    IndentScope.Writer.IndWrite($"binary<ops::swishb>({arguments[0].Name}, tb, {arguments[1].Name});\n");
                }

                break;
            case TIR.CPU.Matmul matmul:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.CPU.Matmul>(matmul)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                }).Result);

                break;
            case TIR.CPU.Pack pack:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.CPU.Pack>(pack)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = string.Join(string.Empty, Enumerable.Repeat(' ', IndentScope.Writer.Indent)),
                }).Result);
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

        symbol = new(
            expr.CheckedType switch
            {
                TensorType t => t.DType.ToC(),
                AnyType => "auto",
                _ => throw new ArgumentOutOfRangeException(nameof(expr)),
            },
            expr.Name + "_" + expr.GlobalVarIndex.ToString());
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBufferRegion(BufferRegion expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var buffer = Visit(expr.Buffer);
        if (expr.Region.AsValueEnumerable().All(r => r is { Start: TensorConst, Stop: TensorConst, Step: TensorConst step } && step.Value.ToScalar<int>() == 1))
        {
            var begins = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Start).Name))}";
            var extents = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Stop).Name))}";
            symbol = new(string.Empty, $"{buffer.Name}.view(fixed_shape<{begins}>{{}}, fixed_shape<{extents}>{{}})");
            _exprMemo.Add(expr, symbol);
        }
        else
        {
            var begins = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Start).Name))}";
            var extents = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Stop - x.Start).Name))}";
            symbol = new(string.Empty, $"{buffer.Name}.view(make_ranked_shape({begins}), make_ranked_shape({extents}))");
            _exprMemo.Add(expr, symbol);
        }

        return symbol;
    }
}
