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
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Diagnostics;

internal sealed record ScriptSymobl(StringBuilder Span, string Name, bool IsRefSymobl) : IPrintSymbol
{
    private int _printCount;

    public ScriptSymobl(StringBuilder span)
        : this(span, string.Empty, false)
    {
    }

    public string Serialize()
    {
        if (IsRefSymobl && _printCount > 0)
        {
            return Name;
        }

        if (IsRefSymobl && _printCount == 0)
        {
            _printCount++;
        }

        return Span.ToString();
    }

    public override string? ToString()
    {
        return Serialize();
    }
}

internal sealed class ScriptPrintContext : IIRPrinterContext
{
    private readonly Dictionary<Expr, ScriptSymobl> _exprMemo;
    private readonly ScriptPrintVisitor _printVisitor;

    public ScriptPrintContext(Dictionary<Expr, ScriptSymobl> exprMemo, ScriptPrintVisitor visitor)
    {
        _exprMemo = exprMemo;
        _printVisitor = visitor;
    }

    public Call? CurrentCall { get; set; }

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

    /// <inheritdoc/>
    public IPrintSymbol Visit(Expr expr) => _printVisitor.Visit(expr);

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
    private readonly Dictionary<Expr, ScriptSymobl> _exprMemo = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<BaseFunction, ScriptSymobl> _extFuncMemo = new(ReferenceEqualityComparer.Instance);
    private readonly bool _displayCallable;

    public ScriptPrintVisitor(TextWriter textWriter, bool display_callable)
    {
        _scope = new(textWriter);
        _context = new(_exprMemo, this);
        _displayCallable = display_callable;
    }

    public ScopeWriter Scope => _scope;

    /// <inheritdoc/>
    public override string VisitType(TensorType type) => type.DType switch
    {
        PrimType ptype => ptype.GetDisplayName() + (type.Shape.IsScalar ? string.Empty : type.Shape.ToString()),
        PointerType { ElemType: PrimType etype } => $"*{etype.GetDisplayName()}",
        ValueType vtype => vtype.GetDisplayName() + (type.Shape.IsScalar ? string.Empty : type.Shape.ToString()),
        _ => throw new NotSupportedException(type.DType.GetType().Name),
    };

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

        var il_sb = new StringBuilder();
        if (_displayCallable)
        {
            var il_visitor = new ILPrintVisitor(new StringWriter(il_sb), false, 0);
            il_visitor.Visit(expr);
        }
        else
        {
            il_sb.Append($"{expr.Name} = Function({VisitType(expr.CheckedType)})");
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

        var il_sb = new StringBuilder();
        if (_displayCallable)
        {
            var il_visitor = new ILPrintVisitor(new StringWriter(il_sb), false, 0);
            il_visitor.Visit(expr);
        }
        else
        {
            il_sb.Append($"{expr.Name} = Fusion<{expr.ModuleKind}>({VisitType(expr.CheckedType)})");
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
        doc = new(_scope.Pop());
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
        _scope.Append($"MemSpan({start}, {size})@{expr.Location}");
        doc = new(_scope.Pop());
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
        doc = new(_scope.Pop());
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
                _scope.Append(CompilerServices.PrintOp(op, _context, false));
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

        doc = new(_scope.Pop());
        _exprMemo.Add(expr, doc);
        return doc;
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
        _scope.AppendLine(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType)).Serialize());

        doc = new(_scope.Pop(), expr.Name, true);
        _exprMemo.Add(expr, doc);

        // 3. only write all doc into root scope
        _scope.AppendLine(doc.Span);

        foreach (var extFunc in _extFuncMemo.Values)
        {
            _scope.IndWriteLine(extFunc.Serialize());
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

        doc = (ScriptSymobl)_scope.GetUniqueVarSymbol(expr);
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
        _scope.Append(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType)).Serialize());

        doc = new(_scope.Pop());
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
                _scope.IndWriteLine(Visit(item).Serialize());
            }
        }

        _scope.IndWrite(")");

        doc = new(_scope.Pop());
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
            _scope.Append(VisitTypeSequential(expr.InitBody, string.Empty).Serialize());
            _scope.Append(".");
        }
        else
        {
            _scope.RemoveLast();
        }

        // 4. wirte body
        _scope.Append("Body");
        _scope.AppendLine(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType)).Serialize());

        doc = new(_scope.Pop());
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
        _scope.Append(VisitTypeSequential(expr.Then, VisitType(expr.CheckedType)).Serialize());

        if (expr.Else.Count > 0)
        {
            _scope.Append(".Then");
            _scope.Append(VisitTypeSequential(expr.Else, string.Empty).Serialize());
        }

        doc = new(_scope.Pop());
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
        _scope.Append(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType), 0).Serialize());

        doc = new(_scope.Pop());
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
        _scope.Append($"T.Buffer({expr.Name}, {VisitType(expr.ElemType)}, {memSpan.Span}, [{string.Join(',', expr.Dimensions.AsValueEnumerable().Select(Visit).Select(e => e.Span.ToString()).ToArray())}], [{string.Join(',', expr.Strides.AsValueEnumerable().Select(Visit).Select(e => e.Span.ToString()).ToArray())}])");
        doc = new(_scope.Pop(), expr.Name, true);
        _exprMemo.Add(expr, doc);
        return doc;
    }

    protected override IPrintSymbol VisitBufferOf(BufferOf expr)
    {
        if (_exprMemo.TryGetValue(expr, out var doc))
        {
            return doc;
        }

        var buffer = Visit(expr.Input);
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
                if (rg.Step is TensorConst con && con.Value.ToScalar<int>() == 1)
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

        doc = new ScriptSymobl(sb, buffer.Name, false);
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
                _scope.IndWriteLine(Visit(item).Serialize());
            }
        }

        _scope.IndWrite(")");

        doc = new(_scope.Pop());
        _exprMemo.Add(expr, doc);
        return doc;
    }
}
