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
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Diagnostics;

internal sealed class PatternPrintVisitor : ExprFunctor<string, string>
{
    private readonly ScopeWriter _scope;
    private readonly Dictionary<Expr, string> _names = new Dictionary<Expr, string>(ReferenceEqualityComparer.Instance);
    private int _localId;

    public PatternPrintVisitor(TextWriter textWriter, int indentLevel)
    {
        _scope = new(textWriter, indentLevel);
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

    /// <inheritdoc/>
    protected override string VisitCall(Call expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var target = Visit(expr.Target);
        var args = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = IsCall(\"{name}\", IsOp<{expr.Target.GetType().Name}>(), IsVArgs({string.Join(",", args)}));\n");

        // AppendCheckedType(expr.CheckedType);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitConst(Const expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = IsTensorConst(\"{name}\");\n");
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitFunction(Function expr)
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

            // _scope.IndWriteLine($"{name} = new Function(\"{expr.Name}\", {body}, new Var[] {{{StringUtility.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit))}}});");
        }

        // 3. Function signature
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

        name = AllocateTempVar(expr);

        _scope.IndWrite($"Fusion {name}");
        AppendCheckedType(expr.CheckedType);
        _scope.Push();
        _scope.IndWriteLine("{");
        using (_scope.IndentUp())
        {
            var body_builder = new StringBuilder();
            string body;
            using (var body_writer = new StringWriter(body_builder))
            {
                var visitor = new PatternPrintVisitor(body_writer, _scope.IndentLevel) { _localId = _localId };
                body = visitor.Visit(expr.Body);
                _scope.Append(body_writer.ToString());
            }

            _scope.IndWriteLine($"{name} = new Fusion(\"{expr.Name}\", \"{expr.ModuleKind}\", {body}, new Var[] {{{StringUtility.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit))}}});");
        }

        _scope.IndWriteLine("}");
        _scope.Append(_scope.Pop());
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitOp(Op expr)
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
    protected override string VisitTuple(IR.Tuple expr)
    {
        if (_names.TryGetValue(expr, out var name))
        {
            return name;
        }

        var fields = expr.Fields.AsValueEnumerable().Select(Visit).ToArray();
        name = AllocateTempVar(expr);
        _scope.IndWrite($"var {name} = IsTuple(\"{name}\", IsVArgs({string.Join(",", fields)}));\n");

        // AppendCheckedType(expr.CheckedType);
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

        name = AllocateTempVar(expr);
        _scope.IndWriteLine($"var {name} = IsWildcard(\"{expr.Name}\");\n");
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitNone(None expr)
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
    protected override string VisitMarker(Marker expr)
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
}
