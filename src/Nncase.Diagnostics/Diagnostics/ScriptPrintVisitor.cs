// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.Math;
using Nncase.IR.Shapes;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Diagnostics;

internal sealed record ScriptSymobl(string Span, string Name, bool IsRefSymobl) : IPrintSymbol
{
    private int _printCount;

    public ScriptSymobl(string span)
        : this(span, string.Empty, false)
    {
    }

    public override string ToString()
    {
        if (IsRefSymobl && _printCount > 0)
        {
            return Name;
        }

        if (IsRefSymobl && _printCount == 0)
        {
            _printCount++;
        }

        return Span;
    }
}

internal sealed class ScriptPrintContext : IPrintOpContext
{
    private readonly Dictionary<BaseExpr, ScriptSymobl> _exprMemo;
    private readonly ScriptPrintVisitor _printVisitor;

    public ScriptPrintContext(Dictionary<BaseExpr, ScriptSymobl> exprMemo, ScriptPrintVisitor visitor, PrinterFlags flags)
    {
        _exprMemo = exprMemo;
        _printVisitor = visitor;
        Flags = flags;
    }

    public Call? CurrentCall { get; set; }

    public PrinterFlags Flags { get; }

    public IPrintSymbol GetArgument(Op op, ParameterInfo parameter)
    {
        if (op.GetType() == parameter.OwnerType)
        {
            return _exprMemo[GetCurrentCall().Arguments[parameter.Index]];
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    public IPrintSymbol[] GetArguments(Op op)
    {
        return (from arg in GetCurrentCall().Arguments.AsValueEnumerable() select _exprMemo[arg]).ToArray();
    }

    /// <inheritdoc/>
    public IPrintSymbol Get(Op op) => _exprMemo[op];

    public string Indent() => new string(' ', _printVisitor.Scope.IndentLevel);

    private Call GetCurrentCall() => CurrentCall ?? throw new InvalidOperationException("Current call is not set.");
}

/// <summary>
/// NOTE:
/// 1. each visit method create a new scope
/// 2. each block expr's start with newline and indent
///
/// <example>
/// `indent` if (x){
/// `indent` &lt;- the current block start from here.
/// `indent` }&lt;- end without new line.
/// </example>
///
/// 3. each block expr's end without newline
/// <example>
/// `indent` if (x){
/// `indent` `indent` x++;
/// `indent` }&lt;- end without new line.
/// </example>
///
/// 4. in block expr, each line expr like const/var write without indent!.
/// </summary>
internal sealed class ScriptPrintVisitor : ExprFunctor<IPrintSymbol, string>
{
    private readonly ScopeWriter _scope;
    private readonly ScriptPrintContext _context;
    private readonly Dictionary<BaseExpr, ScriptSymobl> _exprMemo = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<BaseFunction, ScriptSymobl> _extFuncMemo = new(ReferenceEqualityComparer.Instance);

    public ScriptPrintVisitor(TextWriter textWriter, PrinterFlags flags)
    {
        _scope = new(textWriter);
        _context = new(_exprMemo, this, flags);
        Flags = flags;
    }

    public ScopeWriter Scope => _scope;

    public PrinterFlags Flags { get; }

    /// <inheritdoc/>
    public override string VisitType(TensorType type) => type.DType switch
    {
        PrimType ptype => ptype.GetDisplayName() + (type.Shape.IsScalar ? string.Empty : type.Shape.ToString()),
        PointerType { ElemType: PrimType etype } => $"*{etype.GetDisplayName()}",
        ValueType vtype => vtype.GetDisplayName() + (type.Shape.IsScalar ? string.Empty : type.Shape.ToString()),
        VectorType vtype => $"{vtype.ElemType.GetDisplayName()}<{string.Join(",", vtype.Lanes)}>" + (type.Shape.IsScalar ? string.Empty : type.Shape.ToString()),

        _ => throw new NotSupportedException(type.DType.GetType().Name),
    };

    public override string VisitType(DistributedType type)
    {
        var shape = ((RankedShape)type.TensorType.Shape).ToArray();
        foreach (var (s, r) in type.AxisPolices.Select((s, r) => (s, r)))
        {
            if (s is SBPSplit split)
            {
                if (shape[r].IsFixed)
                {
                    shape[r] = shape[r] / split.Axes.Select(a => type.Placement.Hierarchy[a]).Aggregate(1, (a, b) => a * b);
                }
            }
        }

        var sshape = shape.Select(s => s.ToString()).ToArray();
        foreach (var (s, r) in type.AxisPolices.Select((s, r) => (s, r)))
        {
            if (s is SBPSplit split)
            {
                sshape[r] += string.Join(string.Empty, split.Axes.Select(a => $"@{type.Placement.Name[a]}"));
            }
        }

        return $"Dist({VisitType(type.TensorType)}, ({string.Join(',', type.AxisPolices)}), [{string.Join(',', sshape)}])";
    }

    /// <inheritdoc/>
    public override string VisitType(CallableType type) =>
        $"({string.Join(", ", type.Parameters.Select(VisitType))}) -> {VisitType(type.ReturnType)}";

    /// <inheritdoc/>
    public override string VisitType(TupleType type) =>
        $"({string.Join(", ", type.Fields.Select(VisitType))})";

    /// <inheritdoc/>
    public override string VisitType(InvalidType type) => $"Invalid:{type.Reason}";

    /// <inheritdoc/>
    public override string VisitType(NoneType type) => $"";

    /// <inheritdoc/>
    public override string VisitType(AnyType type) => "any";

    /// <inheritdoc/>
    protected override IPrintSymbol VisitFunction(Function expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        string il_sb;
        if (Flags.HasFlag(PrinterFlags.Detailed))
        {
            il_sb = CompilerServices.Print(expr, Flags & ~PrinterFlags.Script);
        }
        else
        {
            il_sb = $"{expr.Name} = Function({VisitType(expr.CheckedType)})";
        }

        doc = new(il_sb, expr.Name, true);
        _extFuncMemo[expr] = doc;

        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitFusion(Fusion expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        string il_sb;
        if (Flags.HasFlag(PrinterFlags.Detailed))
        {
            il_sb = CompilerServices.Print(expr, Flags & ~PrinterFlags.Script);
        }
        else
        {
            il_sb = $"{expr.Name} = Fusion<{expr.ModuleKind}>({VisitType(expr.CheckedType)})";
        }

        doc = new(il_sb, expr.Name, true);
        _extFuncMemo[expr] = doc;

        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitTuple(IR.Tuple expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _scope.Push();
        _scope.Append($"{{{StringUtility.Join(", ", from item in expr.Fields.AsValueEnumerable() select Visit(item).ToString())}}}");
        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitMemSpan(MemSpan expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        var start = Visit(expr.Start);
        var size = Visit(expr.Size);
        _scope.Push();
        _scope.Append($"MemSpan({start}, {size})@<{expr.Hierarchy}, {expr.Location}>");
        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitMarker(Marker expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        var target = Visit(expr.Target);
        var attr = Visit(expr.Attribute);
        _scope.Push();
        _scope.Append($"{target}@({expr.Name} = {attr})");
        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitAffineDim(IR.Affine.AffineDim expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = new ScriptSymobl($"d{expr.Position}");
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitCall(Call expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        var target = Visit(expr.Target);
        var args = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        _context.CurrentCall = expr;
        _scope.Push();
        switch (expr.Target)
        {
            case Op op:
                _scope.Append(CompilerServices.PrintOp(op, _context) ?? ((IPrintOpContext)_context).GetDefault(op));
                break;
            case Function:
                _scope.Append($"{target.Name}({string.Join(", ", from a in args select a.ToString())})");
                break;
            case TIR.PrimFunction:
                _scope.AppendLine(string.Empty);
                _scope.IndWrite($"{target.Name}({string.Join(", ", from a in args select a.ToString())})");
                break;
            default:
                _scope.Append($"{target}({string.Join(", ", from a in args select a.ToString())})");
                break;
        }

        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitIf(If expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = new(new($"if ..."));
        _exprMemo.Add(expr, doc!);
        return doc!;

        // var target = Visit(expr.Target);
        // var args = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        // _context.CurrentCall = expr;
        // _scope.Push();
        // switch (expr.Target)
        // {
        //     case Op op:
        //         _scope.Append(CompilerServices.PrintOp(op, _context, false));
        //         break;
        //     case Function:
        //         _scope.Append($"{target.Name}({string.Join(", ", from a in args select a.ToString())})");
        //         break;
        //     case TIR.PrimFunction:
        //         _scope.AppendLine(string.Empty);
        //         _scope.IndWrite($"{target.Name}({string.Join(", ", from a in args select a.ToString())})");
        //         break;
        //     default:
        //         _scope.Append($"{target}({string.Join(", ", from a in args select a.ToString())})");
        //         break;
        // }

        // doc = new(_scope.Pop());
        // _exprMemo.Add(expr, doc);
        // return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitTensorConst(TensorConst @const)
    {
        if (_exprMemo.TryGetValue(@const, out var doc))
        {
            return doc;
        }

        if (@const.Value.Shape == Shape.Scalar)
        {
            doc = new(new($"{@const}"));
        }
        else if (@const.Value.ElementType is VectorType vtype)
        {
            doc = new(new($"{vtype.ElemType.GetDisplayName()}<{string.Join(",", vtype.Lanes)}>" + (@const.Value.Shape.IsScalar ? string.Empty : @const.Value.Shape.ToString())));
        }
        else if (@const.Value.ElementType.IsFloat())
        {
            doc = new(new(@const.Value.Length > 8 ? @const.CheckedShape.ToString() : $"{string.Join(",", @const.Value.ToArray<float>())}"));
        }
        else if (@const.Value.ElementType.IsIntegral())
        {
            doc = new(new(@const.Value.Length > 8 ? @const.CheckedShape.ToString() : $"{string.Join(",", @const.Value.ToArray<int>())}"));
        }
        else if (@const.Value.ElementType is PointerType p)
        {
            doc = new(new($"*{p.ElemType.GetDisplayName()}@{@const.Value.Shape}"));
        }

        _exprMemo.Add(@const, doc!);
        return doc!;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitTupleConst(TupleConst tp)
    {
        if (_exprMemo.TryGetValue(tp, out var doc))
        {
            return doc;
        }

        doc = new(new($"{{{string.Join(",", tp.Value.Select(x => Visit(Const.FromValue(x))))}}}"));

        _exprMemo.Add(tp, doc!);
        return doc!;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitPrimFunction(PrimFunction expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _scope.Push();

        // 1. Function signature
        _scope.IndWrite($"T.PrimFunc(\"{expr.Name}\", {string.Join(", ", expr.Parameters.ToArray().Select(Visit))}).Body");

        // 2. Function body
        _scope.AppendLine(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType)).ToString());

        doc = new(_scope.Pop().ToString(), expr.Name, true);
        _exprMemo.Add(expr, doc);

        // 3. only write all doc into root scope
        _scope.AppendLine(doc.Span);

        foreach (var extFunc in _extFuncMemo.Values)
        {
            _scope.IndWriteLine(extFunc.ToString());
        }

        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitOp(Op expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = new(new(expr.GetType().Name));
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitVar(Var expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = (ScriptSymobl)_scope.GetUniqueVarSymbol(expr, "%");
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitFor(For expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        // the for loop will not used by other expression, so we need save the whole `For` il
        _scope.Push();

        // 1. For Loop signature
        var loop_var = Visit(expr.LoopVar);
        _scope.Append($"T.{expr.Mode}(out var {loop_var}, ({Visit(expr.Domain.Start)}, {Visit(expr.Domain.Stop)}, {Visit(expr.Domain.Step)}), out var f{loop_var}).Body");

        // 2. For Body
        _scope.Append(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType)).ToString());

        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitSequential(Sequential expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _scope.Push();

        _scope.AppendLine("(");

        // 1. Foreach Body
        using (_scope.IndentUp())
        {
            foreach (var item in expr.Fields)
            {
                _scope.IndWriteLine(Visit(item).ToString());
            }
        }

        _scope.IndWrite(")");

        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitBlock(Block expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _scope.Push();

        // 1. write head
        _scope.AppendLine($"T.Block(\"{expr.Name}\").");
        _scope.IndWriteLine($"Alloc({StringUtility.Join(",", expr.AllocBuffers.AsValueEnumerable().Select(Visit))}).");
        _scope.IndWriteLine($"Reads({StringUtility.Join(",", expr.Reads.AsValueEnumerable().Select(Visit))}).");
        _scope.IndWriteLine($"Writes({StringUtility.Join(",", expr.Writes.AsValueEnumerable().Select(Visit))}).");
        _scope.IndWriteLine($"Predicate({Visit(expr.Predicate)}).");

        // 2. write iter var bind
        foreach (var iterVar in expr.IterVars)
        {
            string mode_doc = string.Empty;
            switch (iterVar.Mode)
            {
                case IterationMode.DataParallel:
                    mode_doc = "S";
                    break;
                case IterationMode.CommReduce:
                    mode_doc = "R";
                    break;
                case IterationMode.Ordered:
                    mode_doc = "scan";
                    break;
                case IterationMode.Opaque:
                    mode_doc = "opaque";
                    break;
                default:
                    throw new NotSupportedException($"{iterVar.Mode}");
            }

            _scope.IndWriteLine($"Bind(out var {Visit(iterVar)}, ({Visit(iterVar.Dom.Start)}, {Visit(iterVar.Dom.Stop)}, ({Visit(iterVar.Dom.Step)})), IterMode.{iterVar.Mode}, {Visit(iterVar.Value)}).");
        }

        // 3. write init body
        if (expr.InitBody.Count > 0)
        {
            _scope.IndWrite("Init");
            _scope.Append(VisitTypeSequential(expr.InitBody, string.Empty).ToString());
            _scope.Append(".");
        }
        else
        {
            _scope.RemoveLast();
        }

        // 4. wirte body
        _scope.Append("Body");
        _scope.AppendLine(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType)).ToString());

        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitIterVar(IterVar expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = (ScriptSymobl)_scope.GetUniqueVarSymbol(expr.Value, "v");
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitIfThenElse(IfThenElse expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _scope.Push();
        _scope.Append($"T.If({Visit(expr.Condition)}).Then");
        _scope.Append(VisitTypeSequential(expr.Then, VisitType(expr.CheckedType)).ToString());

        if (expr.Else.Count > 0)
        {
            _scope.Append(".Then");
            _scope.Append(VisitTypeSequential(expr.Else, string.Empty).ToString());
        }

        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitLet(Let expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _scope.Push();
        _scope.Append($"T.Let(out var {Visit(expr.Var)}, {Visit(expr.Expression)}).Body");
        _scope.Append(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType), 0).ToString());

        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitBuffer(TIR.Buffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _scope.Push();
        var memSpan = Visit(expr.MemSpan);
        var distributedType = expr.DistributedType == null ? string.Empty : VisitType(expr.DistributedType);
        _scope.Append($"T.Buffer({expr.Name}, {VisitType(expr.ElemType)}, {memSpan.Span}, [{string.Join(',', expr.Dimensions.AsValueEnumerable().Select(Visit).Select(e => e.Span.ToString()).ToArray())}], [{string.Join(',', expr.Strides.AsValueEnumerable().Select(Visit).Select(e => e.Span.ToString()).ToArray())}], {distributedType})");
        doc = new(_scope.Pop().ToString(), expr.Name, true);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitBufferOf(BufferOf expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _ = Visit(expr.Input);
        doc = new ScriptSymobl(new("BufferOf"), "BufferOf", false);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    protected override IPrintSymbol VisitBufferRegion(TIR.BufferRegion expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        var buffer = Visit(expr.Buffer);
        var sb = new StringBuilder();
        sb.Append(buffer.Name);
        if (expr.Region.Length == 0)
        {
            sb.Append("[()]");
        }
        else
        {
            var regions = expr.Region.AsValueEnumerable().Select(rg =>
            {
                if (rg.Step is DimConst con && con.Value == 1)
                {
                    return $"{Visit(rg.Start)}..{Visit(rg.Stop)}";
                }
                else
                {
                    return $"({Visit(rg.Start)}, {Visit(rg.Stop)}, {Visit(rg.Step)})";
                }
            });
            sb.Append($"[{StringUtility.Join(", ", regions)}]");
        }

        doc = new ScriptSymobl(sb.ToString(), buffer.Name, false);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitNone(None expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = new ScriptSymobl(new("None"), "None", false);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitDimension(Dimension expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = new ScriptSymobl(new("Dimension"), "Dimension", false);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitShape(Shape expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = new ScriptSymobl(new("Shape"), "Shape", false);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitPadding(Padding expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = new ScriptSymobl(new("Padding"), "Padding", false);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitPaddings(Paddings expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        doc = new ScriptSymobl(new("Paddings"), "Paddings", false);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    /// <summary>
    /// indent xxxxxx ( // type_info
    /// indent indent xxx
    /// indent indent xxx
    /// indent ).
    /// </summary>
    private IPrintSymbol VisitTypeSequential(Sequential expr, string type_info, int indent = 2)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        _scope.Push();

        if (type_info != string.Empty)
        {
            _scope.AppendLine("( // " + type_info);
        }
        else
        {
            _scope.AppendLine("(");
        }

        // 1. Foreach Body
        using (_scope.IndentUp(indent))
        {
            foreach (var item in expr.Fields)
            {
                _scope.IndWriteLine(Visit(item).ToString());
            }
        }

        _scope.IndWrite(")");

        doc = new(_scope.Pop().ToString());
        _exprMemo.Add(expr, doc);
        return doc;
    }
}
