// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.TIR;

namespace Nncase.IR;

internal sealed record ScriptSymobl(StringBuilder Span, string Name, bool IsRefSymobl) : IPrintSymbol
{
    int PrintCount = 0;
    public ScriptSymobl(StringBuilder Span) : this(Span, string.Empty, false) { }

    public string Serialize()
    {
        if (IsRefSymobl && PrintCount > 0)
            return Name;
        if (IsRefSymobl && PrintCount == 0)
            PrintCount++;
        return Span.ToString();
    }

    public override string? ToString()
    {
        return Serialize();
    }
}

internal sealed class ScriptPrintContext : IIRPrinterContext
{
    readonly Dictionary<Expr, ScriptSymobl> _exprMemo;
    readonly ExprFunctor<IPrintSymbol, string> _printVisitor;

    public ScriptPrintContext(Dictionary<Expr, ScriptSymobl> exprMemo,
    ExprFunctor<IPrintSymbol, string> visitor)
    {
        _exprMemo = exprMemo;
        _printVisitor = visitor;
    }

    public Call? CurrentCall { get; set; }
    private Call GetCurrentCall() => CurrentCall ?? throw new InvalidOperationException("Current call is not set.");

    public IPrintSymbol GetArgument(Op op, ParameterInfo parameter)
    {
        if (op.GetType() == parameter.OwnerType)
        {
            return _exprMemo[GetCurrentCall().Parameters[parameter.Index]];
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    public IPrintSymbol[] GetArguments(Op op)
    {
        return (from arg in GetCurrentCall().Parameters select _exprMemo[arg]).ToArray();
    }

    /// <inheritdoc/>
    public IPrintSymbol Get(Op op) => _exprMemo[op];

    /// <inheritdoc/>
    public IPrintSymbol Visit(Expr expr) => _printVisitor.Visit(expr);
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
    readonly ScopeWriter Scope;
    readonly ScriptPrintContext context;
    readonly Dictionary<Expr, ScriptSymobl> exprMemo = new(ReferenceEqualityComparer.Instance);
    readonly Dictionary<Function, ScriptSymobl> extFuncMemo = new(ReferenceEqualityComparer.Instance);

    public ScriptPrintVisitor(TextWriter textWriter)
    {
        Scope = new(textWriter);
        context = new(exprMemo, this);
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(Function expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }

        var il_sb = new StringBuilder();
        var il_visitor = new ILPrintVisitor(new StringWriter(il_sb));
        il_visitor.Visit(expr);

        doc = new(il_sb, expr.Name, true);
        extFuncMemo[expr] = doc;

        exprMemo.Add(expr, doc);
        return doc;
    }

    public override IPrintSymbol Visit(Tuple expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();
        Scope.Append($"{{{string.Join(", ", (from item in expr select Visit(item).ToString()))}}}");
        doc = new(Scope.Pop());
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(Call expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        var target = Visit(expr.Target);
        var args = expr.Parameters.Select(Visit).ToArray();
        context.CurrentCall = expr;
        Scope.Push();
        switch (expr.Target)
        {
            case Op op:
                Scope.Append(CompilerServices.PrintOp(op, context, false));
                break;
            case Function:
                Scope.Append($"{target.Name}({string.Join(", ", (from a in args select a.ToString()))})");
                break;
            case TIR.PrimFunction:
                Scope.AppendLine("");
                Scope.IndWrite($"{target.Name}({string.Join(", ", (from a in args select a.ToString()))})");
                break;
            default:
                Scope.Append($"{target}({string.Join(", ", (from a in args select a.ToString()))})");
                break;
        }
        doc = new(Scope.Pop());
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(Const expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        if (expr is TensorConst @const)
        {
            if (@const.Value.Shape == Shape.Scalar)
                doc = new(new($"{expr}"));
            else
                if (@const.Value.ElementType.IsFloat())
            {

                doc = new(new($"{string.Join(",", @const.Value.ToArray<float>())}"));
            }
            else if (@const.Value.ElementType.IsIntegral())
            {
                doc = new(new($"{string.Join(",", @const.Value.ToArray<int>())}"));
            }
            else if (@const.Value.ElementType.IsPointer())
            {
                doc = new(new($"{string.Join(",", @const.Value.ToArray<int>().Select(i => "0x" + i.ToString("X")))}"));
            }
        }
        else if (expr is TupleConst tp)
        {
            doc = new(new($"{{{string.Join(",", tp.Fields.Select(Visit))}}}"));
        }
        else
        {
            throw new NotSupportedException();
        }

        exprMemo.Add(expr, doc!);
        return doc!;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(PrimFunction expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();

        // 1. Function signature

        Scope.IndWrite($"T.PrimFunc(\"{expr.Name}\", {string.Join(", ", expr.Parameters.Select(Visit))}).Body");

        // 2. Function body
        Scope.AppendLine(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType!)).Serialize());

        doc = new(Scope.Pop(), expr.Name, true);
        exprMemo.Add(expr, doc);

        // 3. only write all doc into root scope
        Scope.AppendLine(doc.Span);
        foreach (var extFunc in extFuncMemo.Values)
        {
            Scope.IndWriteLine(extFunc.Serialize());
        }
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(Op expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        doc = new(new(expr.GetType().Name));
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(Var expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        doc = (ScriptSymobl)Scope.GetUniqueVarSymbol(expr);
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(For expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }

        // the for loop will not used by other expression, so we need save the whole `For` il
        Scope.Push();

        // 1. For Loop signature
        var loop_var = Visit(expr.LoopVar);
        Scope.Append($"T.{expr.Mode}(out var {loop_var}, ({Visit(expr.Dom.Start)}, {Visit(expr.Dom.Stop)}, {Visit(expr.Dom.Step)}), out var f{loop_var}).Body");

        // 2. For Body
        Scope.Append(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType!)).Serialize());

        doc = new(Scope.Pop());
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <summary>
    /// indent xxxxxx ( // type_info
    /// indent indent xxx
    /// indent indent xxx
    /// indent )
    /// </summary>
    public IPrintSymbol VisitTypeSequential(Sequential expr, string type_info, int indent = 2)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();

        if (type_info != string.Empty)
            Scope.AppendLine("( // " + type_info);
        else
            Scope.AppendLine("(");

        // 1. Foreach Body
        using (Scope.IndentUp(indent))
        {
            foreach (var item in expr.Fields)
            {
                Scope.IndWriteLine(Visit(item).Serialize());
            }
        }
        Scope.IndWrite(")");

        doc = new(Scope.Pop());
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(Sequential expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();

        Scope.AppendLine("(");
        // 1. Foreach Body
        using (Scope.IndentUp())
        {
            foreach (var item in expr.Fields)
            {
                Scope.IndWriteLine(Visit(item).Serialize());
            }
        }
        Scope.IndWrite(")");

        doc = new(Scope.Pop());
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(Block expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();

        // 1. write head
        Scope.AppendLine($"T.Block(\"{expr.Name}\").");

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

            Scope.IndWriteLine($"Bind(out var {Visit(iterVar)}, ({Visit(iterVar.Dom.Start)}, {Visit(iterVar.Dom.Stop)}, ({Visit(iterVar.Dom.Step)})), IterMode.{iterVar.Mode}, {Visit(iterVar.Value)}).");
        }

        // 3. write init body
        if (expr.InitBody.Count > 0)
        {
            Scope.IndWrite("Init");
            Scope.Append(VisitTypeSequential(expr.InitBody, string.Empty).Serialize());
            Scope.Append(".");
        }
        else
        {
            Scope.RemoveLast();
        }

        // 4. wirte body
        Scope.Append("Body");
        Scope.AppendLine(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType!)).Serialize());

        doc = new(Scope.Pop());
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(BufferLoad expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();
        Scope.Append($"{expr.Buffer.Name}[{string.Join(", ", expr.Indices.Select(Visit))}]");
        doc = new(Scope.Pop());
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(BufferStore expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();
        Scope.Append($"{expr.Buffer.Name}[{string.Join(", ", expr.Indices.Select(Visit))}] = {Visit(expr.Value)}");
        doc = new(Scope.Pop());
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(IterVar expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        doc = (ScriptSymobl)Scope.GetUniqueVarSymbol(expr.Value, "v");
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(IfThenElse expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();
        Scope.Append($"T.If({Visit(expr.Condition)}).Then");
        Scope.Append(VisitTypeSequential(expr.Then, VisitType(expr.CheckedType!)).Serialize());

        if (expr.Else.Count > 0)
        {
            Scope.Append(".Then");
            Scope.Append(VisitTypeSequential(expr.Else, string.Empty).Serialize());
        }

        doc = new(Scope.Pop());
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(Let expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();
        Scope.Append($"T.Let(out var {Visit(expr.Var)}, {Visit(expr.Expression)}).Body");
        Scope.Append(VisitTypeSequential(expr.Body, VisitType(expr.CheckedType!), 0).Serialize());

        doc = new(Scope.Pop());
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(TIR.Buffer expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        Scope.Push();
        Scope.Append($"T.Buffer({expr.Name}, {VisitType(expr.ElemType)})");
        doc = new(Scope.Pop(), expr.Name, true);
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override IPrintSymbol Visit(TIR.BufferRegion expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        var buffer = Visit(expr.Buffer);
        var sb = new StringBuilder();
        sb.Append(buffer.Name);
        if (expr.Region.Count == 0)
        {
            sb.Append("[()]");
        }
        else
        {
            var regions = expr.Region.Select(rg =>
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
            sb.Append($"[{string.Join(", ", regions)}]");
        }
        doc = new ScriptSymobl(sb, buffer.Name, false);
        exprMemo.Add(expr, doc);
        return doc;
    }

    public override IPrintSymbol Visit(None expr)
    {
        if (exprMemo.TryGetValue(expr, out var doc)) { return doc; }
        doc = new ScriptSymobl(new("None"), "None", false);
        exprMemo.Add(expr, doc);
        return doc;
    }

    /// <inheritdoc/>
    public override string VisitType(TensorType type) => type.DType switch
    {
        PrimType ptype => ptype.GetDisplayName() + (type.Shape.IsScalar ? "" : type.Shape.ToString()),
        PointerType { ElemType: PrimType etype } ptype => $"*{etype.GetDisplayName()}",
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
}

