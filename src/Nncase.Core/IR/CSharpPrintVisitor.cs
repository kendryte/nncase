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

internal sealed class CSharpPrintVisitor : ExprFunctor<string, string>
{
    private readonly ScopeWriter _scope;
    private readonly Dictionary<Expr, string> _names = new Dictionary<Expr, string>(ReferenceEqualityComparer.Instance);

    private int _localId;

    public CSharpPrintVisitor(TextWriter textWriter, int indent_level)
    {
        _scope = new(textWriter, indent_level);
    }

    /// <inheritdoc/>
    public override string Visit(Call expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var target = Visit(expr.Target);
        var args = expr.Parameters.Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = new Call({target}, new Expr[] {{{string.Join(", ", args)}}})");
        AppendCheckedType(expr.CheckedType);
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Const expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = GetCSharpConst(expr);
        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Function expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = AllocateTempVar(expr);
        _scope.Push();

        // 1. functionv var
        _scope.IndWrite($"Function {name}");
        AppendCheckedType(expr.CheckedType);

        // 2. Function body
        _scope.IndWriteLine("{");
        using (_scope.IndentUp())
        {
            var body = Visit(expr.Body);
            _scope.IndWriteLine($"{name} = new Function(\"{expr.Name}\", new Var[] {{{string.Join(", ", expr.Parameters.Select(Visit))}}});");
        }

        // 3. Function signature
        _scope.IndWriteLine("}");
        _scope.Append(_scope.Pop());
        return name;
    }

    public override string Visit(Fusion expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = AllocateTempVar(expr);

        _scope.IndWrite($"Fusion {name}");
        AppendCheckedType(expr.CheckedType);
        _scope.Push();
        _scope.IndWriteLine("{");
        using (_scope.IndentUp())
        {
            var body_builder = new StringBuilder();
            using (var body_writer = new StringWriter(body_builder))
            {
                var visitor = new CSharpPrintVisitor(body_writer, _scope.IndentLevel).Visit(expr.Body);
                _scope.Append(body_writer.ToString());
            }

            _scope.IndWriteLine($"{name} = new Fusion(\"{expr.Name}\", {expr.ModuleKind}, new Var[] {{{string.Join(", ", expr.Parameters.Select(Visit))}}});");
        }

        _scope.IndWriteLine("}");
        _scope.Append(_scope.Pop());
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Op expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"new {expr.GetType().Name}({expr.DisplayProperty()})";
        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Tuple expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var fields = expr.Fields.Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = new IR.Tuple(new Expr[]{{{string.Join(", ", fields)}}})");
        AppendCheckedType(expr.CheckedType);
        _scope.IndWriteLine();
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Var expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = AllocateTempVar(expr);
        _scope.IndWriteLine($"var {name} = new Var(\"{expr.Name}\", {GetCSharpIRType(expr.TypeAnnotation)});");
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(None expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = $"None.Default";
        _names.Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    public override string Visit(Marker expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var target = Visit(expr.Target);
        var attr = Visit(expr.Attribute);
        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = new Marker(\"{expr.Name}\",{target},{attr})");
        AppendCheckedType(expr.CheckedType);
        return name;
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
        PointerType { ElemType: PrimType etype } ptype => $"*{etype.GetDisplayName()}",
        ValueType => $"{type.DType.ToString()}",
        _ => throw new NotSupportedException(type.DType.GetType().Name),
    };

    /// <inheritdoc/>
    public override string VisitType(TupleType type) =>
        $"({string.Join(", ", type.Fields.Select(VisitType))})";

    private string AllocateTempVar(Expr expr)
    {
        var name = $"v{_localId++}";
        _names.Add(expr, name);
        return name;
    }

    private void AppendCheckedType(IRType? type, string end = "", bool hasNewLine = true)
    {
        if (type is not null)
        {
            if (hasNewLine)
            {
                _scope.AppendLine($"; // {VisitType(type)}{end}");
            }
            else
            {
                _scope.Append($"; // {VisitType(type)}{end}");
            }
        }
        else
        {
            _scope.Append(";\n");
        }
    }

    private string GetCSharpIRType(IRType type) => type switch
    {
        TensorType ttype => $"new TensorType({ttype.DType.GetCSharpName()}, new int[] {string.Join(",", ttype.Shape)})",
        TupleType ttype => $"new TupleType({string.Join(",", ttype.Fields)})",
        AnyType => "AnyType.Default",
        NoneType => "NoneType.Default",
        _ => "AnyType.Default",
    };

    private string GetArrayComma(Shape shape) => string.Join(string.Empty, Enumerable.Repeat<char>(',', shape.Rank - 1));

    private string GetCSharpConst(Const @const) => @const switch
    {
        TensorConst tc => tc.Value.ElementType switch
        {
            PrimType primType => tc.Value.Shape switch
            {
                Shape { IsScalar: true } => tc.Value.GetArrayString(false),
                Shape x when x.Size < 8 => $"new {primType.GetBuiltInName()}[{GetArrayComma(x)}]{tc.Value.GetArrayString(false)}",
                _ => $"Testing.Rand<{primType.GetBuiltInName()}>({string.Join(",", tc.Value.Shape.ToValueArray())})",
            },
            ValueType valueType => $"Tensor.From<QuantParam>(new {valueType.GetBuiltInName()}[{GetArrayComma(tc.Value.Shape)}]{tc.Value.GetArrayString(false)},new[]{{{string.Join(",", tc.Value.Shape)}}})",
            _ => "NotSupport",
        },
        TupleConst tc => $"new TupleConst(new Const[] {{{string.Join(",", tc.Fields.Select(GetCSharpConst))}}})",
        _ => throw new ArgumentOutOfRangeException(@const.GetType().Name),
    };
}
