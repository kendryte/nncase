// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Diagnostics;

/// <summary>
/// a TextWirter, it's have _scope data struct.
/// </summary>
public sealed class ScopeWriter
{
    private readonly TextWriter _rootWriter;

    /// <summary>
    /// stack container.
    /// </summary>
    private readonly Stack<(StringBuilder, TextWriter)> _scopeStack = new();

    /// <summary>
    /// record the all var name's in this scope and parent's scope.
    /// </summary>
    private readonly Dictionary<string, int> _globalVarCountMap = new();

    /// <summary>
    /// the scopes var name stack.
    /// </summary>
    private readonly Stack<List<IPrintSymbol>> _varSymbolStack = new();

    /// <summary>
    /// current writer.
    /// </summary>
    private TextWriter _writer;

    /// <summary>
    /// Initializes a new instance of the <see cref="ScopeWriter"/> class.
    /// ctor.
    /// </summary>
    /// <param name="textWriter">writer.</param>
    /// <param name="indent_level">init indent level.</param>
    public ScopeWriter(TextWriter textWriter, int indent_level = 0)
    {
        IndentLevel = indent_level;
        _rootWriter = textWriter;
        _writer = textWriter;
        _varSymbolStack.Push(new());
    }

    /// <summary>
    /// Gets or sets indent level.
    /// </summary>
    public int IndentLevel { get; set; }

    /// <summary>
    /// Gets current VarNamelist.
    /// </summary>
    private List<IPrintSymbol> VarSymbolList => _varSymbolStack.Peek();

    /// <summary>
    /// push the new string writer, tempoary record the current code into this frame.
    /// </summary>
    public void Push()
    {
        var builder = new StringBuilder();
        TextWriter writer = new StringWriter(builder);
        _scopeStack.Push((builder, writer));
        _writer = writer;

        _varSymbolStack.Push(new());
    }

    /// <summary>
    /// get current frame string.
    /// </summary>
    public StringBuilder Pop()
    {
        var (builder, writer) = _scopeStack.Pop();
        writer.Dispose();
        if (_scopeStack.Count == 0)
        {
            _writer = _rootWriter;
        }
        else
        {
            _writer = _scopeStack.Peek().Item2;
        }

        foreach (var name in _varSymbolStack.Pop())
        {
            _globalVarCountMap[name.Name]--;
            if (_globalVarCountMap[name.Name] == 0)
            {
                _globalVarCountMap.Remove(name.Name);
            }
        }

        // VarNameList
        return builder;
    }

    /// <summary>
    /// insert indent and write.
    /// </summary>
    public void IndWrite(string? value) => Indent().Write(value);

    /// <summary>
    /// write the string builder.
    /// </summary>
    public void IndWrite(StringBuilder? value) => Indent().Write(value);

    /// <summary>
    /// insert indent and write line.
    /// </summary>
    public void IndWriteLine(string? value = null) => Indent().WriteLine(value);

    /// <summary>
    /// wrtie string builder.
    /// </summary>
    public void IndWriteLine(StringBuilder? value) => Indent().WriteLine(value);

    /// <summary>
    /// Append the current line tail, without the indent.
    /// </summary>
    public void Append(string value) => _writer.Write(value);

    /// <summary>
    /// wrtie string builder.
    /// </summary>
    public void Append(StringBuilder value) => _writer.Write(value);

    /// <summary>
    /// Append the current line tail, without the indent, but add new line.
    /// </summary>
    public void AppendLine(string value) => _writer.WriteLine(value);

    /// <summary>
    /// wrtie string builder.
    /// </summary>
    public void AppendLine(StringBuilder value) => _writer.WriteLine(value);

    /// <summary>
    /// remove last char.
    /// </summary>
    public void RemoveLast()
    {
        var sb = _scopeStack.Peek().Item1;
        sb.Remove(sb.Length - 1, 1);
    }

    /// <summary>
    /// add the indent level, return the indent mananger for auto indent down.
    /// </summary>
    public IndentMananger IndentUp(int indent_diff = 2)
    {
        return new(this, indent_diff);
    }

    /// <summary>
    /// get the unique var symbol.
    /// </summary>
    /// <param name="var">var name.</param>
    /// <param name="prefix">prefix name.</param>
    public IPrintSymbol GetUniqueVarSymbol(Var @var, string prefix = "")
    {
        if (!_globalVarCountMap.TryGetValue(prefix + @var.Name + "_" + @var.GlobalVarIndex.ToString(), out var count))
        {
            count = 0;
        }

        var symbol = new ScriptSymobl(new(prefix + @var.Name + "_" + @var.GlobalVarIndex.ToString() + (count == 0 ? string.Empty : $"_{count}")), @var.Name, false);
        count++;
        _globalVarCountMap[@var.Name] = count;
        return symbol;
    }

    /// <summary>
    /// insert the indent.
    /// </summary>
    private TextWriter Indent()
    {
        for (int i = 0; i < IndentLevel; i++)
        {
            _writer.Write(" ");
        }

        return _writer;
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
    private readonly ScopeWriter _parent;

    /// <summary>
    /// the indent add/sub diff value.
    /// </summary>
    private readonly int _indentDiff;

    /// <summary>
    /// Initializes a new instance of the <see cref="IndentMananger"/> class.
    /// <see cref="IndentMananger"/>.
    /// </summary>
    public IndentMananger(ScopeWriter parent, int level_diff = 1)
    {
        _parent = parent;
        _indentDiff = level_diff;
        _parent.IndentLevel += _indentDiff;
    }

    /// <summary>
    /// reduce indentLevel.
    /// </summary>
    public void Dispose()
    {
        _parent.IndentLevel -= _indentDiff;
    }
}

internal sealed class ILPrintVisitor : ExprFunctor<string, string>
{
    private readonly bool _displayCallable;
    private readonly ScopeWriter _scope;
    private readonly Dictionary<Expr, string> _names = new Dictionary<Expr, string>(ReferenceEqualityComparer.Instance);

    private int _localId;

    public ILPrintVisitor(TextWriter textWriter, bool display_callable, int indent_level)
    {
        _displayCallable = display_callable;
        _scope = new(textWriter, indent_level);
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
        PrimType ptype => ptype.GetDisplayName() + (type.Shape.IsScalar ? string.Empty : type.Shape.ToString()),
        PointerType { ElemType: PrimType etype } => $"*{etype.GetDisplayName()}",
        ValueType => $"{type.DType.ToString()}",
        _ => throw new NotSupportedException(type.DType.GetType().Name),
    };

    /// <inheritdoc/>
    public override string VisitType(TupleType type) =>
        $"({string.Join(", ", type.Fields.Select(VisitType))})";

    /// <inheritdoc/>
    protected override string VisitCall(Call expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var target = Visit(expr.Target);
        var property = expr.Target is Op op && op.DisplayProperty() is string prop && prop != string.Empty ? (prop + ", ") : string.Empty;
        var args = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        _scope.IndWrite($"{name} = {target}({property}{string.Join(", ", args)}) {expr.GetHashCode()}");
        AppendCheckedType(expr.CheckedType);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitIf(If expr)
    {
        foreach (var expr1 in expr.ParamList)
        {
            Visit(expr1);
        }

        _scope.IndWriteLine($"if({Visit(expr.Condition)}, Params: ({string.Join(",", expr.ParamList.AsValueEnumerable().Select(Visit))})) " + "{");
        using (_scope.IndentUp())
        {
            Visit(expr.Then);
        }

        _scope.IndWriteLine("} else {");
        using (_scope.IndentUp())
        {
            Visit(expr.Else);
        }

        _scope.IndWriteLine("}");
        return "if";
    }

    /// <inheritdoc/>
    protected override string VisitConst(Const expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        string valueStr = expr switch
        {
            TensorConst tc => tc.Value.Shape.Size <= 8 ? tc.Value.GetArrayString(false) : string.Empty,
            TupleConst => string.Empty,
            _ => throw new ArgumentOutOfRangeException(nameof(expr)),
        };
        valueStr = valueStr != string.Empty ? " : " + valueStr : string.Empty;
        name = $"const({VisitType(expr.CheckedType)}{valueStr})";

        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitFunction(Function expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"%{expr.Name}";
        _names.Add(expr, name);
        _scope.Push();

        // 1. Function signature
        _scope.IndWrite($"{name} = fn({StringUtility.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit))})");
        AppendCheckedType(expr.CheckedType);
        _scope.IndWriteLine("{");

        // 2. Function body
        using (_scope.IndentUp())
        {
            var body = Visit(expr.Body);
        }

        // 3. Function closing
        _scope.IndWriteLine("}");
        _scope.Append(_scope.Pop());
        return name;
    }

    protected override string VisitFusion(Fusion expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"%{expr.Name}";
        _names.Add(expr, name);
        _scope.Push();

        // 1. Function signature
        _scope.IndWrite($"{name} = fusion<{expr.ModuleKind}>({StringUtility.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit))})");
        AppendCheckedType(expr.CheckedType);
        _scope.IndWriteLine("{");

        // 2. Function body
        if (_displayCallable)
        {
            using (_scope.IndentUp())
            {
                var body_builder = new StringBuilder();
                using (var body_writer = new StringWriter(body_builder))
                {
                    var visitor = new ILPrintVisitor(body_writer, true, _scope.IndentLevel).Visit(expr.Body);
                    _scope.Append(body_writer.ToString());
                }
            }
        }
        else
        {
            _scope.IndWriteLine("...");
        }

        // 3. Function closing
        _scope.IndWriteLine("}");
        _scope.Append(_scope.Pop());
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitPrimFunctionWrapper(PrimFunctionWrapper expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"%{expr.Name}";
        _names.Add(expr, name);
        _scope.Push();

        // 1. Function signature
        _scope.IndWrite($"{name} = prim_wrapper({string.Join(", ", expr.ParameterTypes.Select(x => x == null ? string.Empty : VisitType(x)))})");
        AppendCheckedType(expr.CheckedType, " {");

        // 2. Function body
        if (_displayCallable)
        {
            using (_scope.IndentUp())
            {
                using (var bodys = new StringReader(CompilerServices.Print(expr.Target)))
                {
                    while (bodys.ReadLine() is string line)
                    {
                        _scope.IndWriteLine(line);
                    }
                }
            }
        }
        else
        {
            _scope.IndWriteLine("...");
        }

        // 3. Function closing
        _scope.IndWriteLine("}");
        _scope.Append(_scope.Pop());
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitOp(Op expr)
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
    protected override string VisitTuple(IR.Tuple expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var fields = expr.Fields.AsValueEnumerable().Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        _scope.IndWrite($"{name} = ({string.Join(", ", fields)})");
        AppendCheckedType(expr.CheckedType);
        _scope.IndWriteLine();
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitVar(Var expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"%{expr.Name}";
        _names.Add(expr, name);
        if (expr.CheckedType is IRType type)
        {
            name += $": {VisitType(type)}";
        }

        return name;
    }

    /// <inheritdoc/>
    protected override string VisitNone(None expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"None";
        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitMarker(Marker expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var target = Visit(expr.Target);
        var attr = Visit(expr.Attribute);
        name = AllocateTempVar(expr);
        _scope.IndWrite($"{name} = {target}@({expr.Name} = {attr})");
        AppendCheckedType(expr.CheckedType);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitFor(For expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        // the for loop will not used by other expression, so we need save the whole `For` il
        _scope.Push();

        // 1. For Loop signature
        _scope.Append($"For {expr.Mode}({Visit(expr.LoopVar)} in Range({Visit(expr.Domain.Start)}, {Visit(expr.Domain.Stop)}, {Visit(expr.Domain.Step)})");
        AppendCheckedType(expr.CheckedType, " {");

        // 2. For Body
        using (_scope.IndentUp())
        {
            Visit(expr.Body);
        }

        // 3. For closing
        _scope.IndWriteLine("}");

        // 4. extact whole il
        _scope.IndWrite(_scope.Pop());
        return string.Empty;
    }

    /// <inheritdoc/>
    protected override string VisitSequential(Sequential expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        _scope.Push();

        // 1. Sequential signature
        _scope.Append($"Sequential");
        AppendCheckedType(expr.CheckedType, " {", hasNewLine: true);

        // 2. For Body
        using (_scope.IndentUp())
        {
            foreach (var item in expr.Fields)
            {
                Visit(item);
            }
        }

        // 3. For closing
        _scope.IndWriteLine("}");

        // 4. extact whole il
        _scope.IndWrite(_scope.Pop());
        return string.Empty;
    }

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
                _scope.AppendLine($": // {VisitType(type)}{end}");
            }
            else
            {
                _scope.Append($": // {VisitType(type)}{end}");
            }
        }
        else
        {
            _scope.Append("\n");
        }
    }
}
