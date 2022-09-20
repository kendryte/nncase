// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.TIR;

namespace Nncase.IR;

sealed internal class ILPrintVisitor : ExprFunctor<string, string>
{
    private readonly ScopeWriter Scope;
    private readonly Dictionary<Expr, string> _names = new Dictionary<Expr, string>(ReferenceEqualityComparer.Instance);
    bool DisplayCallable;

    private int _localId = 0;

    public ILPrintVisitor(TextWriter textWriter, bool display_callable)
    {
        Scope = new(textWriter);
        DisplayCallable = display_callable;
    }

    /// <inheritdoc/>
    public override string Visit(Call expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }
        var target = Visit(expr.Target);
        var property = expr.Target is Op op && op.DisplayProperty() is string prop && prop != "" ? (prop + ", ") : "";
        var args = expr.Parameters.Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        Scope.IndWrite($"{name} = {target}({property}{string.Join(", ", args)})");
        AppendCheckedType(expr.CheckedType);
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Const expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }

        string valueStr = expr switch
        {
            TensorConst tc => tc.Value.Shape.Size < 8 ? tc.Value.GetArrayString(false) : string.Empty,
            TupleConst tpc => string.Empty,
            _ => throw new ArgumentOutOfRangeException(),
        };
        valueStr = valueStr != string.Empty ? " : " + valueStr : string.Empty;
        name = $"const({(expr.CheckedType is null ? string.Empty : VisitType(expr.CheckedType))}{valueStr})";

        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Function expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }

        name = $"%{expr.Name}";
        _names.Add(expr, name);
        Scope.Push();

        // 1. Function signature
        Scope.IndWrite($"{name} = fn({string.Join(", ", expr.Parameters.Select(Visit))})");
        AppendCheckedType(expr.CheckedType);
        Scope.IndWriteLine("{");

        // 2. Function body
        using (Scope.IndentUp()) { var body = Visit(expr.Body); }

        // 3. Function closing
        Scope.IndWriteLine("}");
        Scope.Append(Scope.Pop());
        return name;
    }

    public override string Visit(Fusion expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }

        name = $"%{expr.Name}";
        _names.Add(expr, name);
        Scope.Push();

        // 1. Function signature
        Scope.IndWrite($"{name} = fusion<{expr.ModuleKind}>({string.Join(", ", expr.Parameters.Select(Visit))})");
        AppendCheckedType(expr.CheckedType);
        Scope.IndWriteLine("{");

        // 2. Function body
        if (DisplayCallable)
            using (Scope.IndentUp()) { var body = Visit(expr.Body); }
        else
            Scope.IndWriteLine("...");

        // 3. Function closing
        Scope.IndWriteLine("}");
        Scope.Append(Scope.Pop());
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(PrimFunctionWrapper expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }

        name = $"%{expr.Name}";
        _names.Add(expr, name);
        Scope.Push();

        // 1. Function signature
        Scope.IndWrite($"{name} = prim_wrapper({string.Join(", ", expr.ParameterTypes.Select(VisitType))})");
        AppendCheckedType(expr.CheckedType, " {");

        // 2. Function body
        if (DisplayCallable)
        {
            using (Scope.IndentUp())
            {
                using (var bodys = new StringReader(CompilerServices.Print(expr.Target)))
                {
                    while (bodys.ReadLine() is string line)
                        Scope.IndWriteLine(line);
                }
            }
        }
        else
            Scope.IndWriteLine("...");

        // 3. Function closing
        Scope.IndWriteLine("}");
        Scope.Append(Scope.Pop());
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Op expr)
    {
        return expr switch
        {
            Unary op => op.UnaryOp.ToString(),
            Binary op => op.BinaryOp.ToString(),
            Compare op => op.CompareOp.ToString(),
            _ => expr.GetType().Name,
        };
    }

    /// <inheritdoc/>
    public override string Visit(Tuple expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }
        var fields = expr.Fields.Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        Scope.IndWrite($"{name} = ({string.Join(", ", fields)})");
        AppendCheckedType(expr.CheckedType);
        Scope.IndWriteLine();
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Var expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }
        name = $"%{expr.Name}";
        _names.Add(expr, name);
        if (expr.CheckedType is IRType type) { name += $": {VisitType(type)}"; }
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(None expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }
        name = $"None";
        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Marker expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }
        var target = Visit(expr.Target);
        var attr = Visit(expr.Attribute);
        name = AllocateTempVar(expr);
        Scope.IndWrite($"{name} = {target}@({expr.Name} = {attr})");
        AppendCheckedType(expr.CheckedType);
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(For expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }

        // the for loop will not used by other expression, so we need save the whole `For` il
        Scope.Push();

        // 1. For Loop signature
        Scope.Append($"For {expr.Mode}({Visit(expr.LoopVar)} in Range({Visit(expr.Domain.Start)}, {Visit(expr.Domain.Stop)}, {Visit(expr.Domain.Step)})");
        AppendCheckedType(expr.CheckedType, " {");

        // 2. For Body
        using (Scope.IndentUp())
        {
            Visit(expr.Body);
        }

        // 3. For closing
        Scope.IndWriteLine("}");

        // 4. extact whole il
        Scope.IndWrite(Scope.Pop());
        return "";
    }

    /// <inheritdoc/>
    public override string Visit(Sequential expr)
    {
        if (_names.TryGetValue(expr, out var name)) { return name; }
        Scope.Push();

        // 1. Sequential signature
        Scope.Append($"Sequential");
        AppendCheckedType(expr.CheckedType, " {", hasNewLine: true);

        // 2. For Body
        using (Scope.IndentUp())
        {
            foreach (var item in expr.Fields) { Visit(item); }
        }

        // 3. For closing
        Scope.IndWriteLine("}");

        // 4. extact whole il
        Scope.IndWrite(Scope.Pop());
        return "";
    }

    /// <inheritdoc/>
    public override string VisitType(AnyType type) => "any";

    /// <inheritdoc/>
    public override string VisitType(CallableType type) =>
        $"({string.Join(", ", type.Parameters.Select(VisitType))}) -> {VisitType(type.ReturnType)}";

    /// <inheritdoc/>
    public override string VisitType(InvalidType type) => $"invalid:{type.Reason}";

    /// <inheritdoc/>
    public override string VisitType(NoneType type) => $"";

    /// <inheritdoc/>
    public override string VisitType(TensorType type) => type.DType switch
    {
        PrimType ptype => ptype.GetDisplayName() + (type.Shape.IsScalar ? "" : type.Shape.ToString()),
        PointerType { ElemType: PrimType etype } ptype => $"*{etype.GetDisplayName()}",
        ValueType => $"{type.DType.ToString()}",
        _ => throw new NotSupportedException(type.DType.GetType().Name),
    };

    /// <inheritdoc/>
    public override string VisitType(TupleType type) =>
        $"({string.Join(", ", type.Fields.Select(VisitType))})";

    private string AllocateTempVar(Expr expr)
    {
        var name = $"%{_localId++}";
        _names.Add(expr, name);
        return name;
    }

    private void AppendCheckedType(IRType? type, string end = "", bool hasNewLine = true)
    {
        if (type is not null)
        {
            if (hasNewLine)
            {
                Scope.AppendLine($": // {VisitType(type)}{end}");
            }
            else
            {
                Scope.Append($": // {VisitType(type)}{end}");
            }
        }
        else
        {
            Scope.Append("\n");
        }
    }
}

/// <summary>
/// a TextWirter, it's have Scope data struct.
/// </summary>
public sealed class ScopeWriter
{
    /// <summary>
    /// current writer.
    /// </summary>
    TextWriter Writer;

    TextWriter rootWriter;

    /// <summary>
    /// current VarNamelist.
    /// </summary>
    List<IPrintSymbol> VarSymbolList => VarSymbolStack.Peek();

    /// <summary>
    /// stack container.
    /// </summary>
    readonly Stack<(StringBuilder, TextWriter)> ScopeStack = new();

    /// <summary>
    /// indent level.
    /// </summary>
    public int indentLevel = 0;

    /// <summary>
    /// record the all var name's in this scope and parent's scope.
    /// </summary>
    readonly Dictionary<string, int> GlobalVarCountMap = new();

    /// <summary>
    /// the scopes var name stack.
    /// </summary>
    readonly Stack<List<IPrintSymbol>> VarSymbolStack = new();

    /// <summary>
    /// ctor.
    /// </summary>
    /// <param name="textWriter"></param>
    public ScopeWriter(TextWriter textWriter)
    {
        rootWriter = textWriter;
        Writer = textWriter;
        VarSymbolStack.Push(new());
    }

    /// <summary>
    /// push the new string writer, tempoary record the current code into this frame.
    /// </summary>
    public void Push()
    {
        StringBuilder builder = new StringBuilder();
        TextWriter writer = new StringWriter(builder);
        ScopeStack.Push((builder, writer));
        Writer = writer;

        VarSymbolStack.Push(new());
    }

    /// <summary>
    /// get current frame string.
    /// </summary>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public StringBuilder Pop()
    {
        var (builder, writer) = ScopeStack.Pop();
        writer.Dispose();
        if (ScopeStack.Count == 0)
        {
            Writer = rootWriter;
        }
        else
        {
            Writer = ScopeStack.Peek().Item2;
        }

        foreach (var name in VarSymbolStack.Pop())
        {
            GlobalVarCountMap[name.Name]--;
            if (GlobalVarCountMap[name.Name] == 0)
                GlobalVarCountMap.Remove(name.Name);
        }

        // VarNameList
        return builder;
    }

    /// <summary>
    /// insert indent and write.
    /// </summary>
    /// <param name="value"></param>
    public void IndWrite(string? value) => Indent().Write(value);

    /// <summary>
    /// write the string builder.
    /// </summary>
    /// <param name="value"></param>
    public void IndWrite(StringBuilder? value) => Indent().Write(value);

    /// <summary>
    /// insert indent and write line.
    /// </summary>
    /// <param name="value"></param>
    public void IndWriteLine(string? value = null) => Indent().WriteLine(value);

    /// <summary>
    /// wrtie string builder.
    /// </summary>
    /// <param name="value"></param>
    public void IndWriteLine(StringBuilder? value) => Indent().WriteLine(value);

    /// <summary>
    /// Append the current line tail, without the indent.
    /// </summary>
    /// <param name="value"></param>
    public void Append(string value) => Writer.Write(value);

    /// <summary>
    /// wrtie string builder.
    /// </summary>
    /// <param name="value"></param>
    public void Append(StringBuilder value) => Writer.Write(value);

    /// <summary>
    /// Append the current line tail, without the indent, but add new line.
    /// </summary>
    /// <param name="value"></param>
    public void AppendLine(string value) => Writer.WriteLine(value);

    /// <summary>
    /// wrtie string builder.
    /// </summary>
    /// <param name="value"></param>
    public void AppendLine(StringBuilder value) => Writer.WriteLine(value);

    /// <summary>
    /// remove last char.
    /// </summary>
    public void RemoveLast()
    {
        var sb = ScopeStack.Peek().Item1;
        sb.Remove(sb.Length - 1, 1);
    }

    /// <summary>
    /// insert the indent.
    /// </summary>
    /// <returns></returns>
    private TextWriter Indent()
    {
        for (int i = 0; i < indentLevel; i++) { Writer.Write(" "); }
        return Writer;
    }

    /// <summary>
    /// add the indent level, return the indent mananger for auto indent down.
    /// </summary>
    /// <param name="indent_diff"></param>
    /// <returns></returns>
    public IndentMananger IndentUp(int indent_diff = 2)
    {
        return new(this, indent_diff);
    }

    /// <summary>
    /// get the unique var symbol.
    /// </summary>
    /// <param name="var">var name</param>
    /// <param name="prefix">prefix name</param>
    /// <returns></returns>
    public IPrintSymbol GetUniqueVarSymbol(Var @var, string prefix = "")
    {
        if (!GlobalVarCountMap.TryGetValue(prefix + @var.Name, out var count))
        {
            count = 0;
        }
        var symbol = new ScriptSymobl(new(prefix + @var.Name + (count == 0 ? "" : $"_{count}")), @var.Name, false);
        count++;
        GlobalVarCountMap[@var.Name] = count;
        return symbol;
    }
}

/// <summary>
/// mananger the wirte indent.
/// </summary>
public sealed class IndentMananger : IDisposable
{
    /// <summary>
    /// the parent scope wirter.
    /// </summary>
    readonly ScopeWriter Parent;

    /// <summary>
    /// the indent add/sub diff value.
    /// </summary>
    readonly int indentDiff;

    /// <summary>
    /// <see cref="IndentMananger"/>.
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="level_diff"></param>
    public IndentMananger(ScopeWriter parent, int level_diff = 1)
    {
        Parent = parent;
        indentDiff = level_diff;
        Parent.indentLevel += indentDiff;
    }

    /// <summary>
    /// reduce indentLevel
    /// </summary>
    public void Dispose()
    {
        Parent.indentLevel -= indentDiff;
    }
}
